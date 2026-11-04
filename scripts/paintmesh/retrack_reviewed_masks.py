#!/usr/bin/env python3
"""Visually prompted SAM + official DeAOT, reviewed before replacing selected masks."""
import argparse
import copy
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO.parent / 'Inpaint360GS/Segment-and-Track-Anything'


def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.now().astimezone().isoformat()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(p)


def record(path):
    path = Path(path).resolve(); st = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return dict(path=str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=h.hexdigest())


def frames(case):
    from PIL import Image
    import numpy as np
    tr = Path(case['tracker']); cams = read(tr/'virtual_cameras.json')['cameras']
    names = [c['image_name']+'.png' for c in cams]
    with zipfile.ZipFile(tr/'images.zip') as z:
        assert sorted(z.namelist()) == sorted(names)
        result = [np.array(Image.open(io.BytesIO(z.read(n))).convert('RGB')) for n in names]
    for f,c in zip(result,cams): assert f.shape[:2] == (c['image_height'],c['image_width'])
    return names,result


def overlay(rgb, mask):
    import numpy as np
    out=rgb.copy();m=mask>0
    out[m]=(out[m]*.60+np.array([40,210,155])*.40).astype(np.uint8)
    return out


def setup():
    os.chdir(OFFICIAL);sys.path.insert(0,str(OFFICIAL))
    import torch
    torch.set_num_threads(4)
    return torch


def seeds(job, cases, keys):
    import numpy as np
    from PIL import Image,ImageDraw
    torch=setup()
    from tool.segmentor import Segmentor
    from model_args import sam_args
    sam=Segmentor(copy.deepcopy(sam_args))
    prompts=read(job/'prompts.json')
    for case in cases:
        key=case['key']
        if key not in keys:continue
        names,rgb=frames(case);folder=job/key/'seeds';folder.mkdir(exist_ok=True)
        entries=prompts[key]['anchors'];tiles=[];records=[]
        for index,entry in sorted(entries.items(),key=lambda kv:int(kv[0])):
            idx=int(index);frame=rgb[idx];h,w=frame.shape[:2]
            union=np.zeros((h,w),dtype=bool);scores_out=[]
            sam.have_embedded=False;sam.interactive_predictor.reset_image()
            if entry.get('empty'):
                pass
            elif entry.get('polygon'):
                im=Image.new('L',(w,h));ImageDraw.Draw(im).polygon([tuple(p) for p in entry['polygon']],fill=1)
                union=np.array(im)>0
            else:
                with torch.inference_mode(),torch.cuda.amp.autocast():
                    sam.set_image(frame)
                    for region in entry['regions']:
                        points=np.array(region['points']);labels=np.array(region['labels']);box=np.array(region['box'])
                        assert len(points)==len(labels) and np.any(labels==1)
                        assert (points[:,0]>=0).all() and (points[:,0]<w).all()
                        assert (points[:,1]>=0).all() and (points[:,1]<h).all()
                        masks,scores,logits=sam.interactive_predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=True)
                        best=int(np.argmax(scores))
                        masks,scores,logits=sam.interactive_predictor.predict(point_coords=points,point_labels=labels,box=box,mask_input=logits[best][None],multimask_output=True)
                        best=int(region.get('candidate',np.argmax(scores)))
                        union|=masks[best];scores_out.append(float(scores[best]))
            assert not union.all()
            assert union.any() or entry.get('empty'),key
            Image.fromarray(union.astype(np.uint8)).save(folder/names[idx])
            Image.fromarray(frame).save(folder/f'{idx:05d}_rgb.png')
            view=Image.fromarray(overlay(frame,union));view.save(folder/f'{idx:05d}_overlay.png')
            im=view.copy();im.thumbnail((700,520));tile=Image.new('RGB',(720,550),'white');tile.paste(im,(0,25));ImageDraw.Draw(tile).text((5,5),f'{key} frame {idx} SAM={scores_out} area={union.sum()}',fill='black');tiles.append(tile)
            records.append(dict(frame=idx,prompts=entry,scores=scores_out,area=int(union.sum()),mask=record(folder/names[idx]),source_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
        sheet=Image.new('RGB',(1440,550*((len(tiles)+1)//2)),'white')
        for j,im in enumerate(tiles):sheet.paste(im,((j%2)*720,(j//2)*550))
        sheet.save(folder/'candidates.jpg',quality=94)
        write(folder/'annotation.json',dict(created_at=now(),method='Visual hole prompts; official SAM ViT-B two-pass box/point refinement; explicitly recorded manual corrections',
              test_rgb_used=False,test_gt_masks_used=False,extra_dilation=0,anchors=records,requires_visual_review=True))
        print('SEED',key,[(r['frame'],r['area']) for r in records],flush=True)


def make_review(folder, case, images, masks):
    import numpy as np
    from PIL import Image,ImageDraw
    review=folder/'review';review.mkdir(exist_ok=True)
    stats=[]
    for start in range(0,len(images),60):
        subset=list(range(start,min(start+60,len(images))))
        sheet=Image.new('RGB',(1800,240*((len(subset)+7)//8)),'white')
        for j,idx in enumerate(subset):
            frame=images[idx];mask=masks[idx]>0;h,w=mask.shape
            ys,xs=np.where(mask)
            if len(xs):
                margin=max(15,int(max(xs.max()-xs.min(),ys.max()-ys.min())*.3))
                box=(max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),min(w,int(xs.max())+margin+1),min(h,int(ys.max())+margin+1))
            else:box=(0,0,w,h)
            tile=Image.new('RGB',(225,240),'white');v=Image.fromarray(overlay(frame,mask));crop=v.crop(box);crop.thumbnail((220,200));tile.paste(crop,(0,25));d=ImageDraw.Draw(tile);d.text((4,4),f'{idx:05d} area={mask.sum()} bbox={box}',fill='black')
            sheet.paste(tile,((j%8)*225,(j//8)*240))
            stats.append(dict(frame=idx,area=int(mask.sum()),fraction=float(mask.mean()),bbox=box))
        sheet.save(review/f'crops_{start:05d}_{subset[-1]:05d}.jpg',quality=95)
    for start in range(0,len(images),12):
        sheet=Image.new('RGB',(1440,4*340),'white')
        for j,idx in enumerate(range(start,min(start+12,len(images)))):
            v=Image.fromarray(overlay(images[idx],masks[idx]));v.thumbnail((475,310));x=(j%3)*480;y=(j//3)*340;sheet.paste(v,(x,y+22));ImageDraw.Draw(sheet).text((x+5,y+3),f'{case["key"]} frame {idx:05d}',fill='black')
        sheet.save(review/f'frames_{start:05d}_{min(start+11,len(images)-1):05d}.jpg',quality=92)
    write(review/'statistics.json',stats)


def track(job, cases, keys):
    import numpy as np
    from PIL import Image
    torch=setup()
    from aot_tracker import get_aot
    from model_args import aot_args
    tracker=get_aot(copy.deepcopy(aot_args))
    for case in cases:
        key=case['key']
        if key not in keys:continue
        folder=job/key;annotation=read(folder/'seeds/annotation.json');names,images=frames(case)
        anchors={r['frame']:np.array(Image.open(folder/'seeds'/names[r['frame']])) for r in annotation['anchors']}
        assert 0 in anchors
        seed_signature=record(folder/'seeds/annotation.json')['sha256']
        receipt=folder/'tracking.json'
        if receipt.exists() and read(receipt).get('seed_sha256')==seed_signature:
            print('SKIP',key,'tracked current seeds',flush=True);continue
        dest=folder/'staged/results/images/images_masks';dest.mkdir(parents=True,exist_ok=True)
        started=time.time_ns();masks=[];tracker.restart();active=False
        write(folder/'status.json',dict(state='tracking',frame=0,total=len(images),pid=os.getpid(),updated_at=now()))
        with torch.inference_mode(),torch.cuda.amp.autocast():
            for idx,frame in enumerate(images):
                if idx in anchors:
                    mask=anchors[idx];tracker.restart();active=bool(mask.any())
                    if active:tracker.add_reference_frame(frame,mask,1,0)
                elif active:
                    pred=tracker.track(frame);tracker.update_memory(pred);mask=pred.squeeze().cpu().numpy().astype(np.uint8)
                else:mask=np.zeros(frame.shape[:2],dtype=np.uint8)
                mask=(mask!=0).astype(np.uint8)
                assert mask.shape==frame.shape[:2] and not mask.all()
                Image.fromarray(mask).save(dest/names[idx]);masks.append(mask)
                if idx%15==0:
                    write(folder/'status.json',dict(state='tracking',frame=idx+1,total=len(images),pid=os.getpid(),updated_at=now()))
                    print('TRACK',key,idx+1,'/',len(images),'area',int(mask.sum()),flush=True)
        make_review(folder,case,images,masks)
        write(receipt,dict(complete=True,started_at_ns=started,finished_at=now(),seed_sha256=seed_signature,
              method='Official r50_deaotl propagation; visually annotated SAM reference frames; no render-difference union or automatic geometry resets',
              anchors=sorted(anchors),frames=len(names),archive=record(Path(case['tracker'])/'images.zip'),
              camera=record(Path(case['tracker'])/'virtual_cameras.json'),
              outputs=[record(dest/n) for n in names],requires_visual_review=True))
        write(folder/'status.json',dict(state='awaiting_visual_review',frame=len(names),total=len(names),updated_at=now()))
        print('DONE',key,'review',folder/'review',flush=True)
        tracker.restart();torch.cuda.empty_cache()


def prepare(job,cases,keys):
    """Export inspectable indexed PNGs and overlays; apply explicitly reviewed cleanup."""
    import numpy as np
    import cv2
    from PIL import Image
    prompts=read(job/'prompts.json')
    for case in cases:
        key=case['key']
        if key not in keys:continue
        folder=job/key;receipt=read(folder/'tracking.json')
        assert not (folder/'published.json').exists(), 'Already published'
        names,images=frames(case);base=folder/'staged/results/images'
        dest=base/'images_masks';views=base/'images_masked_frames';views.mkdir(exist_ok=True)
        masks=[];changes=[]
        for name,frame in zip(names,images):
            m=(np.array(Image.open(dest/name))>0).astype(np.uint8)
            if prompts[key].get('keep_largest_component',False):
                n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
                if n>2:
                    revised=(labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
                    changes.append(dict(frame=name,removed_pixels=int(m.sum()-revised.sum())))
                    backup=folder/'before_component_cleanup';backup.mkdir(exist_ok=True)
                    if not (backup/name).exists():shutil.copy2(dest/name,backup/name)
                    m=revised
            # Pixel labels stay 0/1; the palette makes foreground visible in image viewers.
            im=Image.fromarray(m).convert('P');im.putpalette([0,0,0,255,255,255]+[0]*762);im.save(dest/name)
            Image.fromarray(overlay(frame,m)).save(views/name)
            masks.append(m)
        if changes:make_review(folder,case,images,masks)
        receipt['outputs']=[record(dest/n) for n in names]
        receipt['display_outputs']=str(views)
        receipt['component_cleanup']=dict(keep_largest=prompts[key].get('keep_largest_component',False),changes=changes,
            reason='Only enabled for visually verified single-hole cube sequences; remove disconnected chair/background false positives')
        receipt['prepared_at']=now();write(folder/'tracking.json',receipt)
        print('PREPARED',key,'cleanup_frames',len(changes),flush=True)


def publish(job,cases,keys):
    import numpy as np
    from PIL import Image
    for case in cases:
        key=case['key']
        if key not in keys:continue
        if (job/key/'published.json').exists():
            print('PUBLISHED',key,'already',flush=True);continue
        folder=job/key;tr=Path(case['tracker']);r=read(folder/'tracking.json');review=read(folder/'visual_review.json')
        assert review['accepted'] and review['reviewed_frames']==list(range(case['frame_count']))
        assert review['tracking_sha256']==record(folder/'tracking.json')['sha256']
        assert r['seed_sha256']==record(folder/'seeds/annotation.json')['sha256']
        assert r['archive']==record(tr/'images.zip') and r['camera']==record(tr/'virtual_cameras.json')
        masks=folder/'staged/results/images/images_masks'
        names=read(tr/'tracking_session.json')['expected_masks']
        assert sorted(p.name for p in masks.iterdir())==sorted(names)
        for rec,name in zip(r['outputs'],names):
            assert record(masks/name)==rec
            im=np.array(Image.open(masks/name));assert im.shape==(case['height'],case['width']) and set(np.unique(im)).issubset({0,1})
        # Back up the old masks/metadata once; replace exactly the requested tracker outputs.
        backup=tr/'backup_before_reviewed_retrack_20261103';backup.mkdir(exist_ok=True)
        for name in ['results','review','tracking_session.json','automatic_annotation.json']:
            src=tr/name
            if src.exists() or src.is_symlink():
                assert not (backup/name).exists(),backup/name
                src.rename(backup/name)
        (folder/'staged/results').rename(tr/'results')
        shutil.copytree(folder/'review',tr/'review')
        shutil.copytree(folder/'seeds',tr/'annotations',dirs_exist_ok=True)
        shutil.copyfile(folder/'visual_review.json',tr/'visual_review.json')
        previous=read(backup/'tracking_session.json')
        previous.update(complete=True,status='complete',started_at=now(),started_at_ns=r['started_at_ns'],completed_at=now())
        previous['outputs']=dict(masks=[dict(file=n,**{k:v for k,v in record(tr/'results/images/images_masks'/n).items() if k!='path'}) for n in names])
        write(tr/'tracking_session.json',previous)
        write(tr/'automatic_annotation.json',dict(complete=True,method=r['method'],frame_count=len(names),
              visually_reviewed_all_frames=True,review=review,test_rgb_used=False,test_gt_masks_used=False,
              annotation=str(tr/'annotations/annotation.json'),backup=str(backup),job=str(folder)))
        note=dict(updated_at=now(),tracking_replaced=True,old_tracking_backup=str(backup),
                  downstream_lama_and_training_need_regeneration=True,
                  note='Existing LaMa/model/metrics predate these revised masks; they remain historical outputs. Do not relabel them as retrained results.')
        write(tr/'DOWNSTREAM_REGEN_REQUIRED.json',note)
        write(folder/'published.json',dict(complete=True,published_at=now(),tracker=str(tr),frames=len(names),backup=str(backup)))
        write(folder/'status.json',dict(state='published',total=len(names),updated_at=now()))
        print('PUBLISHED',key,len(names),tr,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seed','track','prepare','publish']);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--keys',nargs='+',default=['all']);p.add_argument('--gpu',default='1');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    job=args.job.resolve();cases=read(job/'cases.json');keys=[c['key'] for c in cases] if args.keys==['all'] else args.keys
    assert set(keys)<=set(c['key'] for c in cases)
    dict(seed=seeds,track=track,prepare=prepare,publish=publish)[args.action](job,cases,keys)


if __name__=='__main__':main()
