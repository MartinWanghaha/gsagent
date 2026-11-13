"""First-stage source identity, full triangle audit, and native pixel guards."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_stage1_is_actual_original_mesh_for_both_modes(tmp_path,monkeypatch):
    from mesh_benchmark import core
    original=tmp_path/'edgs/mesh/ours_30000/tsdf_fusion_post.ply'
    original.parent.mkdir(parents=True);original.write_bytes(b'actual original mesh')
    composition=tmp_path/'recomposition'
    core.write(composition/'bag/composition/scene.json',dict(input_selection=dict(original_mesh=str(original))))
    monkeypatch.setattr(core,'RECOMPOSE',composition)
    for mode in core.STATES:
        assert core.specs_for('bag','paintmesh_stage1',mode)==[dict(path=str(original))]
    original.unlink()
    assert core.specs_for('bag','paintmesh_stage1','recomposition') is None


def test_diagnostic_images_cannot_be_downsampled(tmp_path):
    from audit_paintmesh_blur import decoded
    pixels=np.arange(8*10*3,dtype='uint8').reshape(8,10,3)
    file=tmp_path/'image.png';Image.fromarray(pixels).save(file)
    np.testing.assert_array_equal(decoded(file,pixels.shape),pixels)
    with pytest.raises(ValueError,match='resize is forbidden'):
        decoded(file,(4,5,3))


def test_triangle_diagnostic_processes_all_faces_without_decimation(tmp_path):
    from audit_paintmesh_blur import triangle_footprint
    from recompose.io import save_mesh
    path=tmp_path/'plane.ply'
    vertices=np.array([[-2.,-2,4],[2,-2,4],[-2,2,4],[2,2,4]])
    faces=np.array([[0,1,2],[1,3,2]])
    save_mesh(path,vertices,faces,np.ones_like(vertices))
    camera=dict(name='same.jpg',width=8,height=8,K=[[4.,0,4],[0,4,4],[0,0,1]],w2c=np.eye(4))
    evidence=triangle_footprint(path,camera,np.full((8,8),4.),np.ones((8,8),dtype=bool))
    assert evidence['faces']==evidence['faces_visited']==evidence['visible_triangles']==2
    assert evidence['all_faces_processed']
    assert evidence['pixel_edge']['median']==pytest.approx(4*np.sqrt(2))
    assert evidence['pixel_area']['median']==pytest.approx(8.)


def test_detail_diagnostic_ignores_invalid_pixels():
    from audit_paintmesh_blur import image_evidence
    rng=np.random.default_rng(1)
    gt=rng.integers(0,256,(30,30,3),dtype='uint8');pred=gt.copy()
    valid=np.zeros((30,30),dtype=bool);valid[8:22,8:22]=True
    pred[~valid]=0
    result=image_evidence(pred,gt,valid)
    assert result['PSNR']==120
    assert result['SSIM']==pytest.approx(1.)
    assert result['gradient_energy_ratio']==pytest.approx(1.)
