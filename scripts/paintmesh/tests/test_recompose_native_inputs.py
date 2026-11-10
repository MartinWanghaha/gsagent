"""Native pixels and calibrated pointmaps must reach the official crop together."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.evidence import native_generation_view
from recompose.geometry import RayMesh
from recompose.io import write_json,fingerprint
from recompose.mvsam_worker import validate_generation_inputs


def prepared(tmp_path,chunk):
    w,h=63,41;y,x=np.mgrid[:h,:w]
    rgb=np.stack([(x%2)*255,(y%2)*255,(x*17+y*29)%256],-1).astype('uint8')
    source=tmp_path/'source.png';Image.fromarray(rgb).save(source)
    labels=np.zeros((h,w),'uint8');labels[8:31,12:49]=7
    mask=tmp_path/'labels.png';Image.fromarray(labels).save(mask)
    K=np.array([[48.3,0,29.2],[0,52.1,21.7],[0,0,1.]])
    T=np.eye(4);T[:3,3]=[.2,-.3,.1]
    # World plane z=1.9 has camera z=2.0 for this non-identity extrinsic.
    vertices=np.array([[-4,-4,1.9],[4,-4,1.9],[4,4,1.9],[-4,4,1.9]])
    ray=RayMesh(vertices,np.array([[0,1,2],[0,2,3]],dtype='uint32'))
    camera=dict(image=str(source),native_width=w,native_height=h,native_K=K,w2c=T,
                width=21,height=14,K=K/3)
    out=tmp_path/'evidence';(out/'images').mkdir(parents=True);(out/'object').mkdir()
    info,pm,depth=native_generation_view(camera,mask,7,ray,out,'00000.png',chunk_pixels=chunk)
    np.savez(out/'pointmaps.npz',pointmaps_sam3d=pm[None],intrinsics=K[None],extrinsics=T[None,:3],image_files=['00000.png'])
    write_json(out/'observations.json',dict(generation_input_policy='native_rgb_official_crop',views=[
        dict(name='00000.png',original='source.png',validation=False,image=fingerprint(source),generation=info,w2c=T)]))
    return out,rgb,K,pm,depth


def test_native_rgb_not_downsampled_and_rays_aligned(tmp_path):
    out,rgb,K,pm,depth=prepared(tmp_path,113)
    np.testing.assert_array_equal(np.asarray(Image.open(out/'images/00000.png')),rgb)
    np.testing.assert_array_equal(np.asarray(Image.open(out/'object/00000.png'))[:,:,:3],rgb)
    np.testing.assert_allclose(depth,2.,atol=1e-5)
    y,x=np.mgrid[:41,:63];expected=(np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T)*2
    np.testing.assert_allclose(pm.transpose(1,2,0),expected,atol=1e-5)
    _,_,records=validate_generation_inputs(out)
    assert records[0]['rgb_matches_original'] and records[0]['max_reprojection_error_px']<.001


def test_validation_rejects_upsampled_rgb(tmp_path):
    out,*_=prepared(tmp_path,10000)
    path=out/'images/00000.png'
    Image.open(path).resize((21,14)).resize((63,41)).save(path)
    with pytest.raises(ValueError,match='differs from decoded original'):validate_generation_inputs(out)


def test_validation_rejects_wrong_pointmap_rays(tmp_path):
    out,rgb,K,pm,depth=prepared(tmp_path,10000)
    pm[0]+=.1
    with np.load(out/'pointmaps.npz') as data:values=dict(data)
    values['pointmaps_sam3d']=pm[None];np.savez(out/'pointmaps.npz',**values)
    with pytest.raises(ValueError,match='ray mismatch'):validate_generation_inputs(out)


def test_rejects_previous_low_resolution_policy(tmp_path):
    write_json(tmp_path/'observations.json',dict(views=[]))
    with pytest.raises(ValueError,match='native-resolution'):validate_generation_inputs(tmp_path)
