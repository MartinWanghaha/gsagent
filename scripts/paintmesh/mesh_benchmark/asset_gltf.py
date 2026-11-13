"""Export current canonical asset triangles and native UV bitmaps to GLB.

An asset's vertices already contain its shape transform.  The saved
``baked_shape_transform`` is provenance and must never be applied again here.
Texture pixels and UV coordinates are copied, not baked or resampled.  Export
uses standard glTF float32 geometry/UVs and lossless PNG bitmap attachments.
"""
from pathlib import Path
import os
import tempfile
import numpy as np


def _rgba(colors):
    colors=np.asarray(colors)
    if colors.ndim!=2 or colors.shape[1] not in (3,4) or not np.isfinite(colors).all():
        raise ValueError('Finite RGB or RGBA vertex colors required')
    if np.issubdtype(colors.dtype,np.floating):
        colors=np.rint(np.clip(colors,0,1)*255)
    colors=np.clip(colors,0,255).astype('uint8')
    if colors.shape[1]==3:
        colors=np.column_stack([colors,np.full(len(colors),255,dtype='uint8')])
    return colors


def _texture(array):
    from PIL import Image
    a=np.asarray(array)
    if a.ndim!=3 or a.shape[2] not in (3,4) or not np.isfinite(a).all():
        raise ValueError('Texture must be a finite RGB/RGBA bitmap')
    # The generated asset convention stores RGB8/RGBA8. Refuse silent lossy
    # conversion of float/high-bit-depth sources instead of resampling them.
    if a.dtype!=np.uint8:
        raise ValueError('Native texture bitmap must be uint8; no implicit requantization')
    return Image.fromarray(np.ascontiguousarray(a))


def export_textured_asset(asset_path,destination,transform=None):
    """Export one current ``asset.npz`` as a standalone, textured GLB.

    ``transform`` is an optional world/node 4x4 matrix. It is stored on the root
    scene node, keeping the canonical triangle positions unchanged. Faces are
    grouped by material, preserving original face order within each group.
    Textured faces are expanded at seams so per-face UVs remain exact. A
    ``triangle_material`` value of -1 uses the native vertex colors.

    Returns metadata describing the written file. No source file is modified.
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    asset_path=Path(asset_path);destination=Path(destination)
    if destination.suffix.lower()!='.glb':
        raise ValueError('A .glb destination is required')
    with np.load(asset_path,allow_pickle=False) as data:
        asset={k:data[k] for k in data.files}
    vertices=np.asarray(asset['vertices'],dtype='float64')
    faces=np.asarray(asset['faces'])
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Nonempty finite [N,3] vertices required')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or not np.issubdtype(faces.dtype,np.integer):
        raise ValueError('Nonempty integer [M,3] faces required')
    if faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('Triangle index out of bounds')
    faces=faces.astype('int64',copy=False)
    colors=_rgba(asset.get('colors',np.full((len(vertices),3),.5)))
    if len(colors)!=len(vertices):raise ValueError('Vertex/color length mismatch')
    materials=np.asarray(asset.get('triangle_material',np.full(len(faces),-1,dtype='int32')))
    if materials.shape!=(len(faces),) or not np.issubdtype(materials.dtype,np.integer) or np.any(materials < -1):
        raise ValueError('One integer material id >= -1 per triangle is required')
    textured=materials>=0
    uv=asset.get('triangle_uv')
    if textured.any():
        if uv is None or np.shape(uv)!=(len(faces),3,2) or not np.isfinite(uv).all():
            raise ValueError('Finite [M,3,2] triangle UVs required for textured faces')
    T=np.eye(4) if transform is None else np.asarray(transform,dtype='float64')
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('Finite affine homogeneous transform required')
    if abs(np.linalg.det(T[:3,:3]))<1e-15:
        raise ValueError('Non-singular object transform required')

    scene=trimesh.Scene()
    counts={};factor_by_name={};fallback_names=[]
    for material_id in np.unique(materials):
        mid=int(material_id);selected=np.flatnonzero(materials==mid)
        face_vertices=faces[selected]
        name=f'paintmesh_material_{mid}'
        if mid>=0:
            key=f'texture_{mid}'
            if key not in asset:
                raise ValueError(f'Missing native bitmap: {key}')
            image=_texture(asset[key])
            factor=np.asarray(asset.get(f'texture_factor_{mid}',np.ones(4)),dtype='float64').reshape(-1)
            if factor.shape not in [(3,),(4,)] or not np.isfinite(factor).all() or np.any(factor<0) or np.any(factor>1):
                raise ValueError('Texture factor must be finite RGB/RGBA in [0,1]')
            if len(factor)==3:factor=np.r_[factor,1.]
            mat=PBRMaterial(name=name,baseColorTexture=image,baseColorFactor=factor,
                metallicFactor=0.,roughnessFactor=1.,doubleSided=True,
                alphaMode='BLEND' if image.mode=='RGBA' or factor[3]<1 else 'OPAQUE')
            # NPZ colors on textured surfaces are already derived from the
            # texture in upstream extraction. Exporting them as COLOR_0 would
            # multiply the image by those colors a second time.
            visual=TextureVisuals(uv=np.asarray(uv[selected],dtype='float64').reshape(-1,2),material=mat)
            mesh=trimesh.Trimesh(vertices=vertices[face_vertices].reshape(-1,3),
                faces=np.arange(len(selected)*3,dtype='int64').reshape(-1,3),
                visual=visual,process=False,validate=False)
            factor_by_name[name]=factor.tolist()
        else:
            used,inverse=np.unique(face_vertices,return_inverse=True)
            mesh=trimesh.Trimesh(vertices=vertices[used],faces=inverse.reshape(-1,3),
                vertex_colors=colors[used],process=False,validate=False)
            fallback_names.append(name)
        mesh.metadata.update(source_material_id=mid,source_face_count=len(selected),
            shape_already_baked=True,geometry_simplified=False)
        scene.add_geometry(mesh,node_name=name,geom_name=name,transform=T)
        counts[str(mid)]=int(len(selected))

    def finalize(tree):
        # Trimesh internally stores baseColorFactor as uint8. Restore the exact
        # source float factors in glTF JSON after encoding, before packing GLB.
        for material in tree.get('materials',[]):
            name=material.get('name')
            if name in factor_by_name:
                material.setdefault('pbrMetallicRoughness',{})['baseColorFactor']=factor_by_name[name]
            material.setdefault('extensions',{})['KHR_materials_unlit']={}
            material['doubleSided']=True
        if fallback_names:
            fallback=len(tree.setdefault('materials',[]))
            tree['materials'].append(dict(name='paintmesh_native_vertex_colors',doubleSided=True,
                alphaMode='BLEND' if np.any(colors[:,3]<255) else 'OPAQUE',
                pbrMetallicRoughness=dict(baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=1.),
                extensions={'KHR_materials_unlit':{}}))
            for mesh in tree['meshes']:
                if mesh.get('name') in fallback_names:
                    for primitive in mesh['primitives']:primitive['material']=fallback
        extensions=tree.setdefault('extensionsUsed',[])
        if 'KHR_materials_unlit' not in extensions:extensions.append('KHR_materials_unlit')
        tree.setdefault('extras',{}).update(source_asset=str(asset_path.resolve()),
            shape_transform_policy='vertices are already baked; baked_shape_transform not reapplied',
            texture_policy='native UV and lossless PNG pixels, no bake or resample')

    blob=trimesh.exchange.gltf.export_glb(scene,include_normals=False,
        tree_postprocessor=finalize,extension_webp=False,extension_draco=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+destination.name+'-',dir=destination.parent)
    try:
        with os.fdopen(descriptor,'wb') as file:
            file.write(blob);file.flush();os.fsync(file.fileno())
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return dict(path=str(destination.resolve()),size_bytes=len(blob),face_count=len(faces),
        source_vertex_count=len(vertices),material_face_counts=counts,
        textured_face_count=int(textured.sum()),native_vertex_color_face_count=int((~textured).sum()),
        shape_transform_reapplied=False,geometry_simplified=False,
        native_texture_pixels_preserved=True,transform_stored_on_scene_nodes=True)
