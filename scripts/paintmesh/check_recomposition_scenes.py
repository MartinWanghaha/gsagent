#!/usr/bin/env python3
"""Audit every delivered scene, reraycast geometry, and publish actual diagnostic images."""
import argparse
import json
from pathlib import Path
import subprocess
import struct
import sys
import time
import traceback
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from recompose.io import read_json,write_json,ply,xyz
from recompose.evidence import SELECTION,resolve_scene,binary_mask
from recompose.geometry import transform,check_similarity,camera_rays
from recompose.original_gaussians import movement_transform
from recompose.provenance import Fingerprinter

REPO=Path(__file__).resolve().parents[2]


def mesh_arrays(vertices,faces,colors):
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Empty or nonfinite mesh vertices')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or not np.issubdtype(faces.dtype,np.integer):
        raise ValueError('Mesh has no indexed triangles')
    if faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Triangle index outside vertex array')
    if colors.shape!=(len(vertices),3) or not np.isfinite(colors).all():raise ValueError('Invalid vertex colors')
    tri=vertices[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if area.sum()<=0:raise ValueError('Mesh has no nondegenerate surface')
    return dict(vertices=len(vertices),faces=len(faces),degenerate_triangles=int((area<=1e-14).sum()),
                bbox_min=vertices.min(0),bbox_max=vertices.max(0),surface_area=float(area.sum()))


def check_gltf(path,scene):
    """Read every exported position/index and compare it to the actual source mesh."""
    path=Path(path);doc=read_json(path);buffers={}
    for i,b in enumerate(doc['buffers']):
        p=path.parent/b['uri']
        if p.stat().st_size!=b['byteLength']:raise ValueError(f'glTF buffer length mismatch: {p}')
        buffers[i]=np.memmap(p,mode='r',dtype='uint8')
    def accessor(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        dtype=np.dtype({5126:'<f4',5125:'<u4',5121:'u1'}[a['componentType']])
        width={'SCALAR':1,'VEC3':3,'VEC4':4}[a['type']];stride=v.get('byteStride',width*dtype.itemsize)
        offset=v.get('byteOffset',0)+a.get('byteOffset',0)
        end=offset+(a['count']-1)*stride+width*dtype.itemsize
        if end>v.get('byteOffset',0)+v['byteLength'] or end>len(buffers[v['buffer']]):
            raise ValueError('glTF accessor extends outside its buffer view')
        return np.ndarray((a['count'],width),dtype=dtype,buffer=buffers[v['buffer']],offset=offset,strides=(stride,dtype.itemsize))
    expected={}
    for i,ref in enumerate([scene['background']['mesh']]+scene['background'].get('patches',[])):
        expected['background' if i==0 else f'background_patch_{i}']=(ref,None,np.eye(4))
    for node in scene['objects']:expected['object_'+node['id']]=(node['mesh'],node,np.asarray(node['object_to_world']))
    nodes={n['name']:n for n in doc['nodes']}
    if set(nodes)!=set(expected):raise ValueError('glTF independent object/background nodes differ from scene graph')
    if set(doc['scenes'][doc['scene']]['nodes'])!=set(range(len(nodes))):raise ValueError('glTF scene does not contain every node')
    checked=[]
    for name,(ref,node,T) in expected.items():
        n=nodes[name]
        np.testing.assert_allclose(np.asarray(n['matrix']).reshape(4,4).T,T,atol=1e-12)
        primitive=doc['meshes'][n['mesh']]['primitives'][0]
        positions=accessor(primitive['attributes']['POSITION']);indices=accessor(primitive['indices']).reshape(-1,3)
        colors=accessor(primitive['attributes']['COLOR_0'])
        if colors.shape!=(len(positions),4) or not np.all(colors[:,3]==255):raise ValueError('Invalid glTF vertex color/alpha export')
        if node is not None:
            with np.load(ref['path']) as a:v=a['vertices'];f=a['faces'];c=a['colors']
            if len(positions)!=len(v) or len(indices)!=len(f):raise ValueError('Object glTF topology count mismatch')
            np.testing.assert_array_equal(positions,v.astype('<f4'));np.testing.assert_array_equal(indices,f)
            np.testing.assert_array_equal(colors[:,:3],np.rint(np.clip(c,0,1)*255).astype('uint8'))
        else:
            data=ply(ref['path']);v=data['vertex'].data;f=data['face'].data['vertex_indices']
            if len(positions)!=len(v) or len(indices)!=len(f):raise ValueError('Background was truncated in glTF')
            for lo in range(0,len(v),65536):
                np.testing.assert_array_equal(positions[lo:lo+65536],xyz(v[lo:lo+65536]).astype('<f4'))
                c=np.column_stack([v[n][lo:lo+65536] for n in ['red','green','blue']]) if 'red' in v.dtype.names else np.full((len(v[lo:lo+65536]),3),180,'uint8')
                np.testing.assert_array_equal(colors[lo:lo+65536,:3],c)
            for lo in range(0,len(f),65536):np.testing.assert_array_equal(indices[lo:lo+65536],f[lo:lo+65536])
        checked.append(dict(name=name,vertices=len(positions),triangles=len(indices),transform_verified=True,all_geometry_verified=True))
    return dict(nodes=checked,buffers=len(buffers),background_full_geometry=True)


def check_world_glb(source,destination,T):
    def load(path):
        data=Path(path).read_bytes()
        if struct.unpack_from('<4sII',data)!=(b'glTF',2,len(data)):raise ValueError('Invalid GLB header')
        document=None;chunks=[];offset=12
        while offset<len(data):
            size,kind=struct.unpack_from('<II',data,offset);payload=data[offset+8:offset+8+size];offset+=8+size
            if len(payload)!=size:raise ValueError('Truncated GLB chunk')
            if kind==0x4E4F534A:document=json.loads(payload)
            else:chunks.append((kind,payload))
        if document is None:raise ValueError('Missing GLB JSON')
        return document,chunks
    original,original_chunks=load(source);world,world_chunks=load(destination)
    if original_chunks!=world_chunks:raise ValueError('World GLB changed native geometry/texture payloads')
    expected=json.loads(json.dumps(original));active=expected.get('scene',0)
    children=expected['scenes'][active]['nodes'];root=dict(name='PaintMesh object_to_world',matrix=np.asarray(T).T.ravel().tolist(),children=children)
    expected['scenes'][active]['nodes']=[len(expected['nodes'])];expected['nodes'].append(root)
    if expected!=world:raise ValueError('World GLB hierarchy or placement differs from the scene')
    return dict(native_payload_exact=True,world_transform_verified=True)


def native_inputs(evidence,target):
    from recompose.mvsam_worker import validate_generation_inputs
    meta,views,records=validate_generation_inputs(evidence)
    for view in views:
        g=view['generation'];mask=binary_mask(view['mask']['path'],target,(g['width'],g['height']))
        with Image.open(Path(evidence)/'object'/view['name']) as image:
            np.testing.assert_array_equal(np.asarray(image.getchannel('A'))>0,mask)
    if any(Path(v['original']).name.lower().startswith('test') for v in meta['views']):
        raise ValueError('Removed-state test image was used as an object observation')
    stages=read_json(Path(evidence).parent/'generated/official_preprocessing.json')['views']
    if len(stages)!=2*len(views):raise ValueError('Missing official SS/SLAT preprocessing records')
    for i,record in enumerate(stages):
        g=views[i//2]['generation']
        assert record['stage']==('ss' if i%2==0 else 'slat')
        assert (record['source_width'],record['source_height'])==(g['width'],g['height'])
        assert record['object_tensor_shape']==[1,3,518,518] and record['scene_tensor_shape']==[1,3,518,518]
    return dict(generation_views=len(views),validation_views=sum(v['validation'] for v in meta['views']),
                all_rgb_pixels_original=True,all_alpha_masks_match=True,
                official_object_and_full_scene_tensors_518=True,
                max_reprojection_error_px=max(r['max_reprojection_error_px'] for r in records))


def validation_figure(problem,T,asset,path):
    """Dense holdout projections; show the object mesh itself rather than a GS image."""
    from recompose.texture import sample_colors
    panels=[];metrics=[]
    for view in problem.views:
        if not view['meta']['validation']:continue
        o=view['obs'];truth=o['mask'];h,w=truth.shape;yy,xx=np.mgrid[:h,:w]
        rays=camera_rays(o['K'],o['w2c'],np.c_[xx.ravel(),yy.ravel()]);z,fi,_=problem.mesh.cast(rays,T)
        hit=np.isfinite(z).reshape(h,w);depth=z.reshape(h,w)
        predicted=hit&((o['depth']<=0)|(depth<=o['depth']+problem.cfg['occlusion_tolerance']*problem.d)|truth)
        iou=float((predicted&truth).sum()/max((predicted|truth).sum(),1))
        ys,xs=np.where(truth);margin=max(10,round(max(np.ptp(xs),np.ptp(ys))*.3))
        box=(max(0,int(xs.min()-margin)),max(0,int(ys.min()-margin)),min(w,int(xs.max()+margin+1)),min(h,int(ys.max()+margin+1)))
        rendered=np.full((h*w,3),235,dtype='uint8');ids=np.flatnonzero(predicted.ravel())
        points=rays[ids,:3]+rays[ids,3:]*z[ids,None];local=transform(points,np.linalg.inv(T))
        tri=problem.vertices[problem.faces[fi[ids]]];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];v=local-tri[:,0]
        aa=(a*a).sum(1);ab=(a*b).sum(1);bb=(b*b).sum(1);av=(a*v).sum(1);bv=(b*v).sum(1)
        denominator=np.maximum(aa*bb-ab*ab,1e-30);u=(bb*av-ab*bv)/denominator;v=(aa*bv-ab*av)/denominator
        bary=np.clip(np.c_[1-u-v,u,v],0,1);bary/=np.maximum(bary.sum(1,keepdims=True),1e-30)
        rendered[ids]=np.rint(sample_colors(asset,fi[ids],bary)*255).astype('uint8')
        error=(o['rgb']*.35).astype('uint8');error[truth&predicted]=[50,180,80]
        error[truth&~predicted]=[60,110,255];error[predicted&~truth]=[255,60,50]
        row=Image.new('RGB',(960,285),'white');draw=ImageDraw.Draw(row)
        draw.text((8,5),f'{view["meta"]["original"]} | dense full-mask IoU={iou:.4f}',fill='black')
        for i,(array,title) in enumerate([(o['rgb'],'Observed RGB'),(rendered.reshape(h,w,3),'Generated mesh: baked colors'),(error,'Green overlap / blue missing / red extra')]):
            crop=Image.fromarray(array).crop(box);crop=ImageOps.contain(crop,(312,238),Image.Resampling.LANCZOS)
            row.paste(crop,(320*i+(320-crop.width)//2,42+(238-crop.height)//2));draw.text((320*i+5,25),title,fill='black')
        panels.append(row);metrics.append(dict(view=view['meta']['original'],dense_iou=iou))
    page=Image.new('RGB',(960,36+285*len(panels)),'white');ImageDraw.Draw(page).text((8,10),path.stem+' | independent generation holdout views',fill='black')
    for i,panel in enumerate(panels):page.paste(panel,(0,36+i*285))
    path.parent.mkdir(parents=True,exist_ok=True);page.save(path,quality=94)
    return metrics


def audit_scene(root,scene,out):
    import recompose.scene as scene_module
    import recompose.pipeline as pipeline
    from recompose.optimize import PoseProblem
    from audit_recomposition import compare
    cache=Fingerprinter();scene_module.fingerprint=cache;pipeline.fingerprint=cache
    folder=root/scene;graph=read_json(folder/'composition/scene.json');expected=resolve_scene(REPO/'output/paintmesh/Inpaint360GS',scene)
    report=dict(scene=scene,source=SELECTION[scene],status=graph['status'],checks=[],objects=[])
    def check(name,fn):
        print(scene,name,flush=True)
        try:
            result=fn();report['checks'].append(dict(name=name,passed=True,details=result));return result
        except Exception as error:
            report['checks'].append(dict(name=name,passed=False,error=str(error)));print(traceback.format_exc(),flush=True)
    def source_selection():
        assert graph['input_selection']['group']==SELECTION[scene]
        assert graph['input_selection']['manifest_identity']==expected['manifest_identity']
        assert sorted(n['id'] for n in graph['objects'])==sorted(map(str,expected['target_ids']))
        for kind,key in [('mesh','background_mesh'),('gaussians','background_gaussians')]:
            assert Path(graph['background'][kind]['path']).resolve()==Path(expected[key]).resolve()
        assert graph['components']['gaussians']['training'] is False
        return dict(group=SELECTION[scene],target_ids=expected['target_ids'],training=False)
    check('selected_sources_and_targets',source_selection)
    check('all_scene_reference_hashes',lambda:scene_module.verify_scene_assets(graph))
    def gaussian_records():
        manifest=read_json(folder/'original_gaussians/manifest.json')
        for ref in manifest['sources'].values():assert cache(ref['path'])['sha256']==ref['sha256']
        merged=ply(folder/'composition/world/composed_gaussians.ply')['vertex'].data
        original=ply(manifest['sources']['original']['path'])['vertex'].data;bg=ply(expected['background_gaussians'])['vertex'].data
        start=compare(bg,merged,0);indices_all=[];counts={}
        for node in graph['objects']:
            indices=np.load(node['gaussian_indices']['path']);indices_all.append(indices)
            copy=folder/'composition/world'/f'{node["id"]}_gaussians.ply'
            assert cache(copy)['sha256']==cache(node['gaussians']['path'])['sha256']
            compare(original,ply(copy)['vertex'].data,0,indices)
            count=compare(original,merged,start,indices);start+=count;counts[node['id']]=count
            np.testing.assert_array_equal(movement_transform(node),np.eye(4))
            # A common future world displacement must move both representations equally.
            T=np.asarray(node['object_to_world']);D=np.eye(4);D[:3,3]=[.013,-.017,.011]
            moved=dict(node,object_to_world=D@T);np.testing.assert_allclose(movement_transform(moved),D,atol=1e-12)
        indices=np.concatenate(indices_all)
        assert len(np.unique(indices))==len(indices)==manifest['deleted_count']==manifest['original_count']-manifest['retained_count']
        assert start==len(merged)
        return dict(background_count=len(bg),original_object_counts=counts,total=len(merged),
                    all_render_attributes_exact=True,deleted_rows_assigned_once=True,movement_transform_consistent=True)
    check('exact_original_gaussian_reuse_and_movement',gaussian_records)
    check('full_background_and_independent_gltf_nodes',lambda:check_gltf(folder/'composition/world/mesh_scene.gltf',graph))
    for node in graph['objects']:
        target=int(node['id']);evidence=Path(node['evidence']);asset=np.load(node['mesh']['path']);pose=read_json(node['pose_report']['path']);T=np.asarray(node['object_to_world'])
        item=dict(id=target,reported_status=pose['status'],reported_reasons=pose['reasons'],config=pose['config']);report['objects'].append(item)
        check('original_observation_hashes_'+node['id'],lambda:pipeline.verify_observations(evidence/'observations.json'))
        item['native_inputs']=check('native_rgb_mask_and_pointmap_'+node['id'],lambda:native_inputs(evidence,target))
        def receipts():
            counts={}
            for name in ['prepare','generate','align']:
                receipt=read_json(folder/'objects'/node['id']/'receipts'/f'{name}.json')
                for ref in receipt['outputs']:assert cache(ref['path'])['sha256']==ref['sha256']
                counts[name]=len(receipt['outputs'])
            return counts
        check('stage_output_receipts_'+node['id'],receipts)
        item['mesh']=check('generated_mesh_arrays_'+node['id'],lambda:mesh_arrays(asset['vertices'],asset['faces'],asset['colors']))
        def world_mesh():
            check_similarity(T);np.testing.assert_allclose(T,pose['object_to_world'],rtol=0,atol=1e-12)
            a=ply(folder/'composition/world'/f'{target}_mesh.ply')
            np.testing.assert_array_equal(xyz(a['vertex'].data),transform(asset['vertices'],T).astype('float32'))
            np.testing.assert_array_equal(a['face'].data['vertex_indices'],asset['faces'])
            shape=np.eye(4);shape[:3,:3]=asset['baked_shape_transform']
            check_world_glb(node['textured_mesh']['path'],folder/'composition/world'/f'{target}_textured.glb',T@shape)
            return dict(vertices=len(a['vertex'].data),triangles=len(a['face'].data),world_coordinates_exact=True)
        check('world_mesh_export_'+node['id'],world_mesh)
        def fresh_geometry():
            p=PoseProblem(evidence,node['mesh']['path'],pose['config']);_,views=p.observation(T,validation=True,details=True)
            height=transform(p.samples,T)@p.up;contacts=np.flatnonzero(height<=np.quantile(height,p.cfg['contact_quantile']))
            _,physics=p.physics(T,contacts,return_detail=True)
            for key,value in physics.items():
                if isinstance(value,(float,int)):np.testing.assert_allclose(value,pose['physics'][key],rtol=1e-5,atol=1e-7)
            for current,old in zip(views,pose['validation']):
                assert current['view']==old['view'];np.testing.assert_allclose(current['iou'],old['iou'],atol=1e-8)
            dense=validation_figure(p,T,asset,out/'figures'/f'{scene}_{target}.jpg')
            return dict(validation_iou_median=float(np.median([v['iou'] for v in views])),
                        dense_iou_median=float(np.median([v['dense_iou'] for v in dense])),dense_views=dense,
                        physics=physics,up_confidence=p.meta['up_confidence'],
                        contact_has_penalty_cap=physics['contact_gap_fraction']>=.299,
                        figure=f'figures/{scene}_{target}.jpg')
        item['fresh_geometry']=check('fresh_holdout_raycast_and_contact_'+node['id'],fresh_geometry)
    check('fresh_scene_collision_checks',lambda:scene_module.validate_scene_placement(graph))
    report['integrity_passed']=all(c['passed'] for c in report['checks'])
    report['placement_passed']=graph['status']=='complete'
    report['checked_at']=time.time();write_json(out/'scenes'/f'{scene}.json',report)
    return report


def summarize(root,out):
    scenes=[read_json(out/'scenes'/f'{s}.json') for s in SELECTION]
    for scene in scenes:
        scene['placement_passed']=fresh_placement_passed(scene)
    accepted=[s['scene'] for s in scenes if s['integrity_passed'] and s['placement_passed']]
    bad=[s['scene'] for s in scenes if not s['integrity_passed']]
    doc=['# 全场景重新检测','',f'检测时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。',
         '',f'输入、文件及导出完整性通过：{len(scenes)-len(bad)}/11；当前几何摆放通过：{len(accepted)}/11。',
         '', '重新读取每个场景的实际资产：原始 RGB/alpha/pointmap、源模型与引用 hash、逐高斯属性、完整背景 glTF、独立节点及世界坐标 PLY；重新射线计算留出轮廓与接触/穿透。',
         '', '不改阈值，不改 mesh，不重新训练。留出仅针对物体生成的输入选择，PGSR 深度不是传感器或扫描真值。','',
         '| 场景 | 来源 | 完整性 | 物体 | 摆放 | 抽样 IoU | 全像素 IoU | 接触间距/d | 加权穿透最大/d | 未加权穿透最大/d | 支撑方向置信度 | 图 |',
         '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    figures=[]
    for s in scenes:
        for o in s['objects']:
            g=o.get('fresh_geometry')
            if not g:
                doc.append(f'| {s["scene"]} | {s["source"]} | 失败 | {o["id"]} | 未完成 | - | - | - | - | - | - | - |');continue
            p=g['physics'];gap='支撑未匹配/惩罚上限' if g['contact_has_penalty_cap'] else f'{p["contact_gap_fraction"]:.6f}'
            doc.append(f'| {s["scene"]} | {s["source"]} | {"通过" if s["integrity_passed"] else "失败"} | {o["id"]} | {o["reported_status"]} | {g["validation_iou_median"]:.6f} | {g["dense_iou_median"]:.6f} | {gap} | {p["oriented_penetration_max"]:.6f} | {p["unweighted_penetration_max"]:.6f} | {g["up_confidence"]:.4f} | [图](checks_all_scenes/{g["figure"]}) |')
            figures.append(out/g['figure'])
    doc+=['','阈值：留出抽样 median IoU ≥ 0.75；接触间距 ≤ 0.015d；加权穿透最大值 ≤ 0.02d；未加权穿透及闭合体包含深度 ≤ 0.04d；支撑方向置信度 ≥ 0.6。d 是观测目标范围对角线，不假定米单位。',
          '', '接触值 0.3 含支撑未匹配时的惩罚，不能解释为测量到了 30% 的真实离地距离。穿透只针对有支撑证据的局部三角面；非闭合背景没有全局内外保证。',
          '', '移动检查验证 mesh 与原始高斯使用相同世界位移；实际移动到新位置后仍需重新做支撑/碰撞检测。几何验收不代表生成纹理达到照片质量。',
          '', '图中绿色是轮廓重合，蓝色是生成 mesh 缺失，红色是多余；中央只画真实生成 mesh 的原生颜色。每物体显示全部 3 个留出视角。',
          '', '人工逐图观察：[全场景外观与几何问题](checks_all_scenes/VISUAL_REVIEW.zh-CN.md)。','']
    translations={'held-out silhouette agreement is insufficient':'留出轮廓不足','support contact is uncertain or floating':'接触不确定或浮空',
        'local penetration exceeds tolerance':'局部穿透超限','large intersection with uncertain completed support requires review':'不确定补全表面的原始穿透超限',
        'support direction is uncertain':'支撑方向不可靠','background surface is enclosed deeply inside the generated object':'背景表面被物体深度包裹'}
    for s in scenes:
        if s['placement_passed'] and s['integrity_passed']:continue
        doc+=['',f'## {s["scene"]}','']
        for c in s['checks']:
            if not c['passed']:doc.append(f'- {c["name"]}：{c["error"]}')
        for o in s['objects']:
            if o['reported_reasons']:doc.append(f'- ID {o["id"]}：'+ '；'.join(translations.get(x,x) for x in o['reported_reasons'])+'。')
    for start in range(0,len(figures),5):
        group=figures[start:start+5];page=Image.new('RGB',(960,len(group)*321),'white')
        for i,p in enumerate(group):
            # The first holdout row from each object's detailed figure.
            with Image.open(p) as image:page.paste(image.crop((0,0,960,321)),(0,i*321))
        page.save(out/f'overview_{start//5+1}.jpg',quality=94)
    summary=dict(checked_at=time.time(),scene_count=len(scenes),object_count=sum(len(s['objects']) for s in scenes),
                 integrity_passed=not bad,integrity_failed_scenes=bad,placement_accepted_scenes=accepted,scenes=scenes)
    write_json(out/'report.json',summary);(root/'SCENE_CHECKS.zh-CN.md').write_text('\n'.join(doc)+'\n')
    return summary


def fresh_placement_passed(report):
    """Require recomputed constraints as well as the published acceptance status."""
    if report.get('status')!='complete' or not report.get('integrity_passed'):return False
    collision=next((c.get('details') for c in report['checks'] if c['name']=='fresh_scene_collision_checks' and c['passed']),None)
    if not collision or not collision.get('accepted'):return False
    if not report['objects']:return False
    for item in report['objects']:
        g=item.get('fresh_geometry');cfg=item['config']
        if not g:return False
        p=g['physics']
        values=[g['validation_iou_median'],g['up_confidence'],p['contact_gap_fraction'],p['oriented_penetration_max'],p['unweighted_penetration_max'],p['volumetric_penetration_max']]
        if not np.isfinite(values).all():return False
        if g['validation_iou_median']<cfg['minimum_iou'] or g['up_confidence']<cfg['minimum_up_confidence']:return False
        if (p['contact_gap_fraction']>cfg['maximum_contact_gap'] or p['oriented_penetration_max']>cfg['maximum_penetration'] or
            p['unweighted_penetration_max']>2*cfg['maximum_penetration'] or p['volumetric_penetration_max']>2*cfg['maximum_penetration']):return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(REPO/'output/paintmesh/Inpaint360GS/recomposition'))
    p.add_argument('--scene',choices=list(SELECTION));a=p.parse_args();root=Path(a.root).resolve();out=root/'checks_all_scenes';out.mkdir(exist_ok=True)
    if a.scene:
        report=audit_scene(root,a.scene,out);return int(not report['integrity_passed'])
    for scene in SELECTION:
        print('Checking',scene,flush=True);log=out/'logs'/f'{scene}.log';log.parent.mkdir(exist_ok=True)
        started=time.time()
        with log.open('w') as f:
            rc=subprocess.call([sys.executable,__file__,'--root',str(root),'--scene',scene],stdout=f,stderr=subprocess.STDOUT)
        result_path=out/'scenes'/f'{scene}.json'
        if not result_path.exists() or result_path.stat().st_mtime<started:
            write_json(out/'scenes'/f'{scene}.json',dict(scene=scene,source=SELECTION[scene],status='audit_failed',integrity_passed=False,placement_passed=False,objects=[],
                checks=[dict(name='audit_process',passed=False,error=f'No fresh report (exit {rc}); see checks_all_scenes/logs/{scene}.log')]))
    result=summarize(root,out);print(json.dumps({k:v for k,v in result.items() if k!='scenes'}),flush=True)
    return int(not result['integrity_passed'])


if __name__=='__main__':raise SystemExit(main())
