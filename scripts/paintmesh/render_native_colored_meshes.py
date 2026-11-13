#!/usr/bin/env python3
"""Render actual colored triangles using the mesh backends shipped by each project."""
import argparse
import ast
from pathlib import Path
import os
import sys
import types
import time
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,cameras,read,write,identity,digest
from mesh_benchmark.color_export import read_mesh,positions
from mesh_benchmark.layout import work_root
from recompose.io import fingerprint

RESEARCH=Path('/home/member/wyc/mesh_gs_research')
WRAP=RESEARCH/'method/meshgs_wrapping_fusion_20261104/upstream/gaussian_wrapping'


def exact_triangle_indices(faces,vertex_count):
    """glDrawElements needs exact integers, including indices above 2**24."""
    indices=np.asarray(faces)
    if indices.ndim!=2 or indices.shape[1]!=3 or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Expected integer triangle indices')
    if indices.size and (indices.min()<0 or indices.max()>=vertex_count or indices.max()>np.iinfo('uint32').max):
        raise ValueError('Triangle index is outside the vertex buffer')
    return np.ascontiguousarray(indices,dtype='uint32')


def projection(camera):
    K=np.asarray(camera['K']);w,h=camera['width'],camera['height'];near,far=.001,1e6
    P=np.zeros((4,4),dtype='float32');P[0,0]=2*K[0,0]/w;P[1,1]=2*K[1,1]/h
    P[0,2]=2*K[0,2]/w-1;P[1,2]=2*K[1,2]/h-1
    P[2,2]=far/(far-near);P[2,3]=-far*near/(far-near);P[3,2]=1
    return P


def upstream_mesh_module(method):
    """Keep upstream renderer classes; supply their camera/geometry interfaces."""
    import torch
    from typing import List,Union,Tuple,Optional
    source=(RESEARCH/'baselines/CoMe' if method=='come' else WRAP)/'scene/mesh.py'
    code=source.read_text().replace('from scene.cameras import Camera','')
    module=types.ModuleType('native_'+method+'_mesh');namespace=module.__dict__
    namespace.update(Camera=types.SimpleNamespace,__file__=str(source))
    if method=='gaussian_wrapping':
        dependency=WRAP/'utils/geometry_utils.py';tree=ast.parse(dependency.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transform_points_world_to_view')
        function.decorator_list=[]
        scope=dict(torch=torch,np=np,List=List,Union=Union,Tuple=Tuple,Optional=Optional,Camera=types.SimpleNamespace)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(dependency),'exec'),scope)
        namespace['transform_points_world_to_view']=scope['transform_points_world_to_view']
        code=code.replace('from utils.geometry_utils import transform_points_world_to_view','')
    # Upstream assumes every raster has a background pixel. Full scene meshes
    # can cover every pixel, so retain index zero when -1 was not present.
    old='filtered_face_idx = filtered_face_idx[1:]'
    if old in code:
        code=code.replace(old,'has_background = bool(filtered_face_idx[0] < 0)\n            filtered_face_idx = filtered_face_idx[1:] if has_background else filtered_face_idx')
        code=code.replace('filtered_pix_to_face = filtered_pix_to_face - 1','filtered_pix_to_face = filtered_pix_to_face - int(has_background)')
    exec(compile(code,str(source),'exec'),namespace)
    return module,source


class NVDMesh:
    def __init__(self,method,vertices,faces,colors):
        import torch
        self.torch=torch;module,self.source=upstream_mesh_module(method);self.module=module
        self.mesh=module.Meshes(verts=torch.from_numpy(vertices.astype('float32')).cuda(),
             faces=torch.from_numpy(faces.astype('int32')).cuda(),verts_colors=torch.from_numpy(colors.astype('float32')/255).cuda())
        self.renderer=None;self.triangle_batch=2**20
    def render(self,camera):
        torch=self.torch;W=np.asarray(camera['w2c']);P=projection(camera)
        c=types.SimpleNamespace(image_width=camera['width'],image_height=camera['height'],
            world_view_transform=torch.tensor(W.T,dtype=torch.float32,device='cuda'),
            full_proj_transform=torch.tensor((P@W).T,dtype=torch.float32,device='cuda'))
        if self.renderer is None:
            self.renderer=self.module.ScalableMeshRenderer(self.module.MeshRasterizer(cameras=[c],use_opengl=False))
        with torch.no_grad():
            while True:
                try:
                    package=self.renderer(self.mesh,cameras=[c],cam_idx=0,return_depth=True,return_normals=False,
                         use_antialiasing=False,return_pix_to_face=True,max_triangles_in_batch=self.triangle_batch)
                    break
                except RuntimeError as error:
                    if 'subtriangle count overflow' not in str(error) or self.triangle_batch<=8192:raise
                    self.triangle_batch//=2;torch.cuda.empty_cache()
                    print('NATIVE_TRIANGLE_BATCH_RETRY',self.triangle_batch,flush=True)
        depth=package['depth'][0,:,:,0].cpu().numpy()
        rgb=package['rgb'][0].clamp(0,1).cpu().numpy()
        valid=(package['pix_to_face'].reshape(camera['height'],camera['width'])>=0).cpu().numpy()&np.isfinite(depth)&(depth>0)
        depth=np.where(valid,depth,0).astype('float32');rgb[~valid]=0
        return np.rint(rgb*255).astype('uint8'),depth


class PyRenderMesh:
    def __init__(self,method,vertices,faces,colors):
        os.environ['PYOPENGL_PLATFORM']='egl'
        os.environ['EGL_DEVICE_ID']=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        sys.path.insert(0,str(BASE/'_scripts/native_mesh_render_deps'))
        import pyrender
        self.p=pyrender
        self.source=(RESEARCH/'baselines/2DGS/scripts/eval_tnt/cull_mesh.py' if method=='2dgs' else RESEARCH/'baselines/GOF/eval_tnt/cull_mesh.py')
        class ExactIndexPrimitive(pyrender.Primitive):
            # pyrender 0.1.45 casts indices to float32 before uploading uint32,
            # silently rounding large vertex IDs into different triangles.
            @pyrender.Primitive.indices.setter
            def indices(self,value):
                self._indices=None if value is None else exact_triangle_indices(value,len(self.positions))
        primitive=ExactIndexPrimitive(positions=vertices.astype('float32'),indices=faces,
             color_0=colors.astype('float32')/255,material=pyrender.MetallicRoughnessMaterial(
                 baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=1.,doubleSided=True))
        self.scene=pyrender.Scene(bg_color=[0,0,0,0]);self.scene.add(pyrender.Mesh(primitives=[primitive]));self.renderer=None
    def render(self,camera):
        p=self.p;K=np.asarray(camera['K']);w,h=camera['width'],camera['height']
        c=p.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.001,zfar=1e6)
        pose=np.linalg.inv(camera['w2c']);pose[:3,1:3]*=-1
        node=self.scene.add(c,pose=pose)
        if self.renderer is None:self.renderer=p.OffscreenRenderer(w,h)
        self.renderer.viewport_width=w;self.renderer.viewport_height=h
        rgb,depth=self.renderer.render(self.scene,flags=p.RenderFlags.FLAT|p.RenderFlags.SKIP_CULL_FACES)
        self.scene.remove_node(node)
        return rgb[:,:,:3],depth.astype('float32')


def render_scene(output,method,scene,limit=None):
    import torch
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    root=work_root(output);marker=read(BASE/method/scene/'mesh_complete.json');source=Path(marker.get('colored_path',marker['path']))
    cs=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cameras(scene) if c['part']=='held_out']
    if limit:cs=cs[:limit]
    dest=root/'native_renderers'/method/scene;dest.mkdir(parents=True,exist_ok=True)
    upstream=(RESEARCH/'baselines/CoMe/scene/mesh.py' if method=='come' else WRAP/'scene/mesh.py' if method=='gaussian_wrapping' else
              RESEARCH/'baselines/2DGS/scripts/eval_tnt/cull_mesh.py' if method=='2dgs' else RESEARCH/'baselines/GOF/eval_tnt/cull_mesh.py')
    request=dict(mesh=fingerprint(source),cameras=cs,upstream_source=fingerprint(upstream),backend='NVDiffRast upstream ScalableMeshRenderer' if method in ['come','gaussian_wrapping'] else 'upstream TNT pyrender backend with flat RGB attachment',
                 adapter=fingerprint(Path(__file__)),appearance='Original vertex RGB, no learned appearance field',gaussian_renderer_used=False)
    signature=digest(request);complete=dest/'complete.json'
    if complete.exists() and read(complete)['signature']==signature:return
    mesh=read_mesh(source);vertices=positions(mesh['vertex'].data)
    faces=np.asarray(mesh['face'].data['vertex_indices']);colors=np.column_stack([mesh['vertex'].data[n] for n in ['red','green','blue']])
    renderer=(NVDMesh if method in ['come','gaussian_wrapping'] else PyRenderMesh)(method,vertices,faces,colors)
    assert request['upstream_source']==fingerprint(renderer.source)
    for c in cs:
        start=time.time();rgb,depth=renderer.render(c)
        (dest/'renders').mkdir(exist_ok=True);(dest/'depth').mkdir(exist_ok=True)
        Image.fromarray(rgb).save(dest/'renders'/(c['stem']+'.png'));np.save(dest/'depth'/(c['stem']+'.npy'),depth)
        print('NATIVE_MESH_RGB',method,scene,c['name'],'coverage',float((depth>0).mean()),'seconds',time.time()-start,flush=True)
    write(complete,dict(complete=True,signature=signature,request=request,views=len(cs),pilot=limit is not None,
          triangle_batch=getattr(renderer,'triangle_batch',None),all_source_triangles_preserved=True,
          integer_index_precision_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(BASE/'mesh_comparison_whole'))
    p.add_argument('--method',choices=['2dgs','come','gof','gaussian_wrapping'],required=True);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args();render_scene(a.output,a.method,a.scene,a.limit)
