#!/usr/bin/env python3
"""Stage isolated low-PSNR repairs and run native pipelines on GPU 1.

Never mutates completed baseline outputs, test masks, or the running GPU-0 job.
Aura's initializer sanitation is explicitly a corrected variant, not an
unmodified official reproduction. No test image participates in a correction.
"""
from pathlib import Path
import argparse
import fcntl
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path('/home/member/wyc')
AUDIT = BASE/'gsagent/logs/low_psnr_audit_20261026'
OFFICIAL = BASE/'Inpaint360GS'
AURA = BASE/'AuraFusion360_official'
CAR_ROOT = OFFICIAL/'output/low_psnr_repair_20261026'
CAR_WORK = CAR_ROOT/'scenes/car/workspace'
AURA_WORK = AURA/'output/low_psnr_repair_20261026/workspace'
PM = BASE/'gsagent'
PM_OLD = PM/'output/paintmesh/Inpaint360GS/inpaint360/cube/removal/full_r4_adaptive180_20261025_105925'
PM_REMOVAL = PM_OLD.parent/'full_r4_adaptive180_holealpha95_20261026'
PM_INPAINT = PM_REMOVAL/'inpaint/full_r4_adaptive180_holealpha95_20261026'
PY = Path('/home/member/.conda/envs')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def read(path): return json.loads(path.read_text())
def now(): return datetime.now().astimezone().isoformat()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
    elif dst.resolve()!=src.resolve():raise ValueError('Unexpected existing path: '+str(dst))

def car_masks(work, output):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw
    original=OFFICIAL/'output/paper_r4/scenes/car/workspace'
    virtual=work/'output/inpaint360/car/virtual/ours_2000'
    original_raw=original/'Segment-and-Track-Anything/tracking_results/images/images_masks'
    output.mkdir(parents=True,exist_ok=True)
    rows=[];pages=[]
    for i in range(30):
        name=f'{i:05}.png'
        label=np.array(Image.open(virtual/'objects_pred'/name))
        if label.ndim!=2:raise ValueError('Expected raw semantic IDs')
        # ID 8 also labels most of the road: never add it to the removal target.
        # Preserve the reviewed car-hole track; only correct its shadow border.
        old=np.array(Image.open(original_raw/name).convert('L'))>127
        target=old
        grown=cv2.dilate(target.astype('uint8'),np.ones((61,61),np.uint8))>0
        protected=cv2.dilate(np.isin(label,[2,4,5,6]).astype('uint8'),np.ones((25,25),np.uint8))>0
        mask=grown&~protected
        final=cv2.dilate(mask.astype('uint8'),np.ones((21,21),np.uint8))>0
        if (final&np.isin(label,[2,4,5,6])).any():raise ValueError('Mask would remove a protected cone')
        if not .01<mask.mean()<.3:raise ValueError('Unreasonable car hole mask')
        Image.fromarray(mask.astype('uint8')*255).save(output/name)
        rgb=np.array(Image.open(virtual/'renders'/name).convert('RGB'))
        rgb[final]=(rgb[final]*.5+[0,127,0]).astype('uint8')
        if i%15==0:pages.append(Image.new('RGB',(1500,900),'#222'))
        yy,xx=np.where(final);crop=Image.fromarray(rgb).crop((max(0,xx.min()-25),max(0,yy.min()-25),min(rgb.shape[1],xx.max()+26),min(rgb.shape[0],yy.max()+26)))
        crop.thumbnail((295,270));x=i%5*300;y=(i%15)//5*300;pages[-1].paste(crop,(x,y+25));ImageDraw.Draw(pages[-1]).text((x,y),f'{name} final {final.mean():.1%}',fill='white')
        rows.append(dict(frame=name,raw_area=float(mask.mean()),prepared_area=float(final.mean()),sha256=sha(output/name)))
    for i,p in enumerate(pages):p.save(output/f'protected_review_{i}.jpg')
    save(output/'manifest.json',dict(scene='car',target_ids=[7],protected_ids=[2,4,5,6],uses_test_GT=False,
        source='own 30 virtual-view semantic projections + previously tracked car masks',
        additional_radius_pixels=30,cone_protection_margin=12,native_dilation_pixels=10,frames=rows))

def prepare_car():
    paper=module('paper_repair',OFFICIAL/'scripts/reproduce_paper.py')
    paper.prepare(CAR_ROOT,['car'])
    old=OFFICIAL/'output/paper_r4/scenes/car/workspace'
    data=CAR_WORK/'data/inpaint360/car';model=CAR_WORK/'output/inpaint360/car';model.mkdir(parents=True,exist_ok=True)
    for name in ('associated_hqsam','associated_hqsam_color','raw_hqsam'):
        link(old/'data/inpaint360/car'/name,data/name)
    for name in ('point_cloud','3dgs_output'):
        link(old/'output/inpaint360/car'/name,model/name)
    link(old/'output/inpaint360/car/train/ours_2000',model/'train/ours_2000')
    link(old/'output/inpaint360/car/test/ours_2000',model/'test/ours_2000')
    shutil.copy2(old/'output/inpaint360/car/cfg_args',model/'cfg_args')
    for family in ('object_removal','object_inpaint'):
        cfg=read(old/f'config/{family}/inpaint360/car.json')
        cfg.update(select_obj_id=[7],target_id=[7])
        save(CAR_WORK/f'config/{family}/inpaint360/car.json',cfg)
    save(CAR_ROOT/'repair_plan.json',dict(created_at=now(),source=str(old),gpu=1,changes=[
         'extend shadow/hole annotation by 30 pixels; protect cones by IDs 2,4,5,6'],
         resolution=4,iterations=5000,losses='unchanged native Inpaint360GS',circle_radius=.9033,
         no_test_GT_used=True,native_sha256={n:sha(OFFICIAL/n) for n in ['edit_object_removal.py','edit_object_inpaint.py','edit_object_removal_plyfusion.py','tools/prepare_lama_data.py']}))

def sanitize_aura(old, initial):
    """Standard one-pass statistical outlier removal on NEW centers only."""
    import numpy as np
    from scipy.spatial import cKDTree
    if len(initial)<=len(old):raise ValueError('Initializer did not add any points')
    if not all(np.array_equal(old[n],initial[n][:len(old)]) for n in old.dtype.names):
        raise ValueError('Background changed before initialization')
    new=initial[len(old):].copy();xyz=np.stack([new[n] for n in ('x','y','z')],axis=1)
    finite=np.isfinite(xyz).all(1)
    if finite.sum()<21:raise ValueError('Too few finite new centers')
    distances=cKDTree(xyz[finite]).query(xyz[finite],k=21)[0][:,1:].mean(1)
    threshold=float(distances.mean()+2*distances.std())
    keep=np.zeros(len(new),dtype=bool);keep[np.flatnonzero(finite)]=distances<=threshold
    if keep.mean()<.9:raise ValueError('Severe initialization failure: manual review required')
    selected=new[keep].copy()
    # Native logit(1) == +inf, so sigmoid derivative is zero forever.
    # Preserve near-opaque intent, but make NEW logits finite and trainable.
    invalid=~np.isfinite(selected['opacity'])
    if np.any(selected['opacity'][invalid]<0):raise ValueError('Unexpected negative infinite opacity')
    selected['opacity'][invalid]=np.log((1-1e-4)/1e-4)
    result=np.concatenate([old,selected])
    if not all(np.array_equal(old[n],result[n][:len(old)]) for n in old.dtype.names):raise AssertionError('Background modified')
    if not all(np.isfinite(selected[n]).all() for n in selected.dtype.names):raise ValueError('Remaining nonfinite initializer property')
    return result,dict(old_count=len(old),original_new=len(new),retained_new=int(keep.sum()),removed_new_indices=np.flatnonzero(~keep).tolist(),
        sor_neighbors=20,sor_std_ratio=2,threshold=threshold,finite_opacity_repairs=int(invalid.sum()),opacity_alpha=1-1e-4,
        old_background_bitwise_unchanged=True,selection_uses='new centers only, no test views/masks/metrics')

def prepare_aura():
    from plyfile import PlyData,PlyElement
    data=AURA_WORK/'data/Other-360/inpaint360_redbull';model=AURA_WORK/'output/Other-360/inpaint360_redbull'
    od=AURA/'data/Other-360/inpaint360_redbull';om=AURA/'output/Other-360/inpaint360_redbull'
    for p in (data,model):p.mkdir(parents=True,exist_ok=True)
    for name in ('images','test_images','sparse','object_masks','unseen_masks','reference','unseen_mask'):
        link(od/name,data/name)
    shutil.copy2(od/'dataset_manifest.json',data/'dataset_manifest.json')
    if not (data/'unseen_masks_dilated').exists():shutil.copytree(od/'unseen_masks_dilated',data/'unseen_masks_dilated')
    for iteration in ('iteration_30000','iteration_30000_object_removal'):
        link(om/'point_cloud'/iteration,model/'point_cloud'/iteration)
    cfg=(om/'cfg_args').read_text().replace(str(od),str(data)).replace(str(om),str(model))
    (model/'cfg_args').write_text(cfg)
    config=AURA_WORK/'configs';config.mkdir(exist_ok=True)
    for name in ('inpaint','sdedit'):
        value=(AURA/f'configs/Inpaint360GS/redbull/{name}.config').read_text().replace(str(od),str(data)).replace(str(om),str(model))
        (config/(name+'.config')).write_text(value)
    dest=model/'point_cloud/iteration_object_inpaint_init/point_cloud.ply'
    if not dest.exists():
        old_path=om/'point_cloud/iteration_30000_object_removal/point_cloud.ply';init_path=om/'point_cloud/iteration_object_inpaint_init/point_cloud.ply'
        old=PlyData.read(old_path)['vertex'].data;initial=PlyData.read(init_path)['vertex'].data
        fixed,stats=sanitize_aura(old,initial)
        dest.parent.mkdir(parents=True,exist_ok=True);PlyData([PlyElement.describe(fixed,'vertex')],text=False).write(dest)
        save(AURA_WORK/'sanitation.json',dict(stats,source_initial_sha256=sha(init_path),source_background_sha256=sha(old_path),corrected_sha256=sha(dest)))
    save(AURA_WORK/'repair_plan.json',dict(created_at=now(),gpu=1,scene='redbull',resolution=4,finetune_iterations=10000,
        changes='new initializer points: one-pass SOR (20 neighbors, 2 sigma), replace +inf opacity logit by logit(0.9999)',
        label='AuraFusion360 initializer-corrected variant; not unmodified official result',
        unchanged='native RGB/loss/density control, old Gaussians, reference image, masks, other hyperparameters',
        no_test_GT_used=True,native_sha256={n:sha(AURA/n) for n in ['inpaint.py','scene/gaussian_model.py','utils/LeftRefill/sdedit_utils.py']}))

def coverage_hole(mask,alpha):
    import numpy as np
    from scipy import ndimage as ndi
    if mask.shape!=alpha.shape or not np.isfinite(alpha).all():raise ValueError('Invalid alpha/mask pair')
    # Keep only the tracked removal footprint with insufficient remaining coverage.
    result=np.asarray(mask,dtype=bool)&(alpha<.95)
    result=ndi.binary_closing(result,iterations=2)
    labels,count=ndi.label(result);counts=np.bincount(labels.ravel())
    if count==0:raise ValueError('No connected removal hole')
    # cube is one object with one contiguous unseen footprint. Remote low-alpha
    # chair/background islands in the old track must not become new holes.
    largest=1+int(np.argmax(counts[1:]));result=ndi.binary_fill_holes(labels==largest)
    if not 50<=result.sum()<mask.size*.3:raise ValueError('Unusable coverage-derived hole')
    return result

def prepare_paintmesh():
    import numpy as np
    from PIL import Image,ImageDraw
    for name in ('work_model','removal_manifest.json','removed_3dgs','removed_mesh','config'):
        link(PM_OLD/name,PM_REMOVAL/name)
    for name in ('virtual_cameras.json','images.zip'):
        link(PM_OLD/'tracker'/name,PM_REMOVAL/'tracker'/name)
    dest=PM_REMOVAL/'tracker/results/images/images_masks';dest.mkdir(parents=True,exist_ok=True)
    review=PM_REMOVAL/'tracker/review';review.mkdir(exist_ok=True)
    session=read(PM_OLD/'tracker/tracking_session.json')
    started=time.time_ns();records=[];stats=[];pages=[]
    for i,name in enumerate(session['expected_masks']):
        old=np.asarray(Image.open(PM_OLD/'tracker/results/images/images_masks'/name))>0
        alpha=np.load(PM_OLD/f'work_model/virtual/ours_object_removal/iteration_2000/alpha/{Path(name).stem}.npy').squeeze()
        mask=coverage_hole(old,alpha)
        out=dest/name;Image.fromarray(mask.astype('uint8')).save(out);s=out.stat()
        records.append(dict(file=name,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(out)))
        stats.append(dict(frame=name,old_fraction=float(old.mean()),new_fraction=float(mask.mean())))
        rgb=np.array(Image.open(PM_OLD/'work_model/virtual/ours_object_removal/iteration_2000/renders'/name).convert('RGB'))
        rgb[mask]=(rgb[mask]*.5+[0,127,0]).astype('uint8')
        if i%30==0:pages.append(Image.new('RGB',(1500,1200),'#222'))
        im=Image.fromarray(rgb);im.thumbnail((300,175));x=i%5*300;y=(i%30)//5*200;pages[-1].paste(im,(x,y+22));ImageDraw.Draw(pages[-1]).text((x,y),name+f' {mask.mean():.1%}',fill='white')
    for i,page in enumerate(pages):page.save(review/f'alpha95_{i}.jpg')
    session.update(complete=True,status='complete',started_at=now(),started_at_ns=started,completed_at=now(),outputs=dict(masks=records),
        annotation_correction=dict(source_session=str(PM_OLD/'tracker/tracking_session.json'),source_sha256=sha(PM_OLD/'tracker/tracking_session.json'),
            method='existing DeAOT tracks intersected with remaining PGSR alpha < 0.95; closing 2, largest connected cube hole >=50 pixels, fill holes',uses_test_GT=False))
    save(PM_REMOVAL/'tracker/tracking_session.json',session)
    save(PM_REMOVAL/'tracker/alpha_correction.json',dict(complete=True,visually_reviewed_all_frames=False,statistics=stats,
        mask_dilation_unchanged=20,reason='full original object silhouette overmasked already reconstructed tabletop'))
    save(AUDIT/'paintmesh_cube_repair_plan.json',dict(gpu=1,removal_root=str(PM_REMOVAL),inpaint_root=str(PM_INPAINT),changes='correct unseen hole masks only',
        resolution=4,views=180,camera='same adaptive PGSR poses',init_depth=True,init_normal=False,surface_init='none',iterations=10000,
        evaluate_iterations=[5000,10000],mask_dilation=20,recursive_guide=False,density='prune only',background='local editable, far strictly frozen',no_test_GT_used=True))

def run_paintmesh():
    review=read(PM_REMOVAL/'tracker/alpha_correction.json')
    if not review['visually_reviewed_all_frames']:raise ValueError('Review all 180 corrected cube masks first')
    raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(PM/'scripts/paintmesh/configs/inpaint360_recommended.env')],cwd=PM,env=dict(os.environ,PAINTMESH_INIT_MODE='mixed'))
    env=dict(entry.decode().split('=',1) for entry in raw.split(b'\0') if b'=' in entry)
    env.update(GPU='1',CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',DATA_ROOT=str(BASE/'data'),OUTPUT_ROOT=str(PM/'output'),
        REMOVAL_ROOT=str(PM_REMOVAL),INPAINT_RUN_ROOT=str(PM_INPAINT),RUN_NAME=PM_REMOVAL.name,INPAINT_RUN_NAME=PM_INPAINT.name,
        MASK_DILATION='20',EDGS_SURFACE_INIT_MODE='none',EDGS_JOINT_SAVE_ITERATIONS='5000',EDGS_INPAINT_CONFIG=str(PM/'scripts/paintmesh/configs/inpaint360_balanced_10000.yaml'))
    env.pop('PYTHONPATH',None)
    conda=shutil.which('conda')
    if not conda:
        conda=next((str(p) for p in [Path('/home/member/miniconda3/bin/conda'),Path('/home/member/.conda/bin/conda')] if p.is_file()),None)
    if not conda:raise ValueError('conda not found')
    env['CONDA_BIN']=conda
    script=['bash','scripts/paintmesh/run_inpaint.sh','Inpaint360GS/inpaint360','cube','4','25','none']
    execute('paintmesh_cube','train_10000',script+['1'],PM,dict(env,END_STAGE='5'))
    execute('paintmesh_cube','snapshot_5000',[PY/'paintmesh/bin/python',Path(__file__),'pm_snapshot'],PM,env)
    for step in (5000,10000):
        if step==10000:execute('paintmesh_cube','publish_10000',script+['6'],PM,dict(env,END_STAGE='6'))
        model=PM_INPAINT/('inpainted_3dgs_5000' if step==5000 else 'inpainted_3dgs')
        execute('paintmesh_cube',f'evaluate_{step}',[PY/'paintmesh/bin/python','scripts/paintmesh/eval_inpaint360.py','run',
            '--model-path',model,'--source-path',BASE/'data/Inpaint360GS/inpaint360/cube','--resolution','4','--iteration',str(step),'--batch-size','4'],PM,env)
    execute('paintmesh_cube','mesh',script+['7'],PM,dict(env,END_STAGE='8'))

def pm_snapshot():
    sys.path.insert(0,str(PM/'scripts/paintmesh'))
    import run_inpaint360_full as full
    full.export_snapshot(PM_INPAINT,5000)

def execute(case,stage,command,cwd,extra=None):
    receipt=AUDIT/'repair_receipts'/case/(stage+'.json')
    if receipt.exists():return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if extra:env.update(extra)
    log=AUDIT/f'{case}_{stage}.log'
    state=dict(state='running',case=case,stage=stage,gpu=1,command=list(map(str,command)),cwd=str(cwd),log=str(log),time=now())
    save(AUDIT/'remaining_repair_status.json',state);print(state,flush=True)
    with log.open('ab',buffering=0) as f:
        p=subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(AUDIT/'remaining_repair_status.json',dict(state,child_pid=p.pid))
        code=p.wait()
    if code:raise RuntimeError(f'{case}/{stage} exited {code}: {log}')
    save(receipt,dict(state,state='complete',completed_at=now()))
    refresh_report()

def run_car():
    # Native removal renders held-out original cameras and normalizes depth
    # against their original semantic render, as it already does for train.
    link(OFFICIAL/'output/paper_r4/scenes/car/workspace/output/inpaint360/car/test/ours_2000',
         CAR_WORK/'output/inpaint360/car/test/ours_2000')
    paper=module('paper_commands',OFFICIAL/'scripts/reproduce_paper.py')
    for phase in ('remove','inpaint','evaluate'):
        if phase=='inpaint':
            car_masks(CAR_WORK,CAR_WORK/'Segment-and-Track-Anything/tracking_results/images/images_masks')
        for stage,python,relative,cmd in paper.commands('car',phase):
            if stage=='virtual_views':cmd+=['--circle_radius','0.9033']
            cwd=CAR_WORK/relative
            # Match reproduce_paper.Runner.execute: LaMa is its own package root.
            extra=dict(PYTHONPATH=str(cwd) if cwd.name=='LaMa' else os.pathsep.join([str(CAR_WORK),str(CAR_WORK/'seg/detectron2')]),
                LAMA_ENV='inpaint360gs-lama',CUDA_HOME=str(paper.MAIN.parent.parent))
            if cwd.name=='LaMa':extra['TORCH_HOME']=str(cwd)
            execute('official_car',stage,[python,*cmd],cwd,extra)

def prepare_aura_sdedit(data,model):
    """Keep the native one-mask-per-training-image input contract on a rerun.

    Finetuning may save an additional PNG mask for the reference image whereas
    initialization saved JPG masks. Archive only demonstrably equivalent extras
    in this isolated workspace; never alter the baseline or choose by test GT.
    """
    import numpy as np
    from PIL import Image
    expected=sorted(read(data/'dataset_manifest.json')['training_images'])
    images=data/'images';masks=data/'unseen_masks_dilated'
    renders=model/'train/ours_object_inpaint_init/renders'
    if sorted(p.name for p in images.iterdir())!=expected:
        raise ValueError('SDEdit training images differ from dataset manifest')
    if sorted(p.name for p in renders.iterdir())!=[f'{i:05}.png' for i in range(len(expected))]:
        raise ValueError('SDEdit initialization renders are incomplete or unordered')
    if not all((masks/name).is_file() for name in expected):
        raise ValueError('Missing native SDEdit training mask')
    by_stem={Path(name).stem:name for name in expected}
    if len(by_stem)!=len(expected):raise ValueError('Ambiguous training image stems')
    planned=[]
    for p in masks.iterdir():
        if p.name in expected:continue
        canonical=by_stem.get(p.stem)
        if not canonical or not p.is_file():raise ValueError('Unexpected SDEdit mask: '+str(p))
        with Image.open(p) as image:a=np.asarray(image.convert('L'))>127
        with Image.open(masks/canonical) as image:b=np.asarray(image.convert('L'))>127
        if not np.array_equal(a,b):raise ValueError('Conflicting duplicate SDEdit mask: '+str(p))
        planned.append((p,canonical))
    for i,name in enumerate(expected):
        sizes=[]
        for p in (images/name,masks/name,renders/f'{i:05}.png'):
            with Image.open(p) as image:sizes.append(image.size)
        if len(set(sizes))!=1:raise ValueError('SDEdit image/mask dimensions differ: '+name)
    archive=data/'sdedit_duplicate_mask_archive';records=[]
    for p,canonical in planned:
        target=archive/p.name;archive.mkdir(exist_ok=True)
        if target.exists():raise ValueError('Duplicate mask archive already occupied: '+str(target))
        records.append(dict(duplicate=p.name,canonical=canonical,sha256=sha(p),binary_masks_equal=True))
        p.rename(target)
    proof=data/'sdedit_input_validation.json'
    prior=read(proof) if proof.exists() else {}
    save(proof,dict(validated_at=now(),training_frames=len(expected),no_test_GT_used=True,
        archived_duplicates=prior.get('archived_duplicates',[])+records,
        masks=[dict(file=name,sha256=sha(masks/name)) for name in expected]))

def run_aura():
    py=PY/'aurafusion360/bin/python';data=AURA_WORK/'data/Other-360/inpaint360_redbull';model=AURA_WORK/'output/Other-360/inpaint360_redbull'
    extra=dict(PYTHONPATH=os.pathsep.join([str(AURA),str(AURA/'utils/LeftRefill')]),
        HF_HUB_CACHE=str(AURA/'ckpt/huggingface/hub'),HUGGINGFACE_HUB_CACHE=str(AURA/'ckpt/huggingface/hub'),
        TORCH_HOME=str(AURA/'ckpt/torch'),XDG_CACHE_HOME=str(AURA/'ckpt'),HF_HUB_OFFLINE='1')
    execute('aura_redbull','render_corrected_init',[py,AURA/'render.py','-s',data,'-m',model,'--iteration','object_inpaint_init','--skip_mesh'],AURA_WORK,extra)
    if not (AUDIT/'repair_receipts/aura_redbull/sdedit.json').exists():prepare_aura_sdedit(data,model)
    execute('aura_redbull','sdedit',[py,AURA/'utils/LeftRefill/sdedit_utils.py','--config',AURA_WORK/'configs/sdedit.config'],AURA_WORK,extra)
    execute('aura_redbull','finetune',[py,AURA/'inpaint.py','--config',AURA_WORK/'configs/inpaint.config','--reference_index','0','--skip_builtin_evaluation',
        '--images','inpaint','--finetune_iteration','10000'],AURA_WORK,extra)
    execute('aura_redbull','collect',[py,Path(__file__),'aura_collect'],AURA_WORK,extra)
    execute('aura_redbull','evaluate',[py,AURA/'scripts/vendor/metrics_inpaint360.py','-m','output/inpaint360/redbull','--crop','1.0','--resize','1.0'],AURA_WORK,extra)

def aura_collect():
    m=module('aura_eval_repair',AURA/'scripts/evaluate_inpaint360.py');m.ROOT=AURA_WORK;m.collect('redbull')
    link(AURA_WORK/'data/Other-360/inpaint360_redbull',AURA_WORK/'data/inpaint360/redbull')

def refresh_report():
    try:
        module('low_psnr_report',Path(__file__).with_name('report_low_psnr_repairs.py')).report()
    except Exception:
        # A reporting failure must never stop an independent training stage.
        traceback.print_exc()

def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    # comm is parenthesized and may contain spaces; fields below start at state.
    fields=text.rsplit(')',1)[1].split()
    return fields[19] if fields[0]!='Z' else None

def queue(wait_pid):
    lock=(AUDIT/'remaining_repair.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Wait for this exact pre-existing GPU-1 repair, without occupying CUDA.
    identity=process_identity(wait_pid)
    while identity and process_identity(wait_pid)==identity:
        save(AUDIT/'remaining_repair_status.json',dict(state='waiting_for_gpu1',wait_pid=wait_pid,queued=['paintmesh_cube','official_car','aura_redbull'],time=now()))
        refresh_report()
        time.sleep(15)
    failures={}
    failure_path=AUDIT/'repair_failures.json'
    if failure_path.exists() and read(failure_path):
        history_path=AUDIT/'repair_failure_history.json'
        history=read(history_path) if history_path.exists() else []
        save(history_path,history+[dict(archived_at=now(),failures=read(failure_path))])
    save(failure_path,failures)
    for name,fn in [('paintmesh_cube',run_paintmesh),('official_car',run_car),('aura_redbull',run_aura)]:
        try:fn()
        except Exception as exc:
            failures[name]=str(exc);traceback.print_exc();save(AUDIT/'repair_failures.json',failures)
    save(failure_path,failures)
    save(AUDIT/'remaining_repair_status.json',dict(state='complete' if not failures else 'completed_with_failures',gpu=1,failures=failures,time=now()))
    refresh_report()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','prepare_pm','pm','pm_snapshot','car','aura','aura_collect','queue']);p.add_argument('--wait-pid',type=int,default=0);args=p.parse_args()
    if args.action=='prepare':prepare_car();prepare_aura()
    elif args.action=='car':run_car()
    elif args.action=='aura':run_aura()
    elif args.action=='aura_collect':aura_collect()
    elif args.action=='prepare_pm':prepare_paintmesh()
    elif args.action=='pm':run_paintmesh()
    elif args.action=='pm_snapshot':pm_snapshot()
    else:queue(args.wait_pid)
