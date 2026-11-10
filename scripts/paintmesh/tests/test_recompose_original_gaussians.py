"""Original splats must remain exact and move with the generated mesh, without fitting."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.gaussian import write_gaussians
from recompose.io import ply,save_mesh,write_json,fingerprint
from recompose.geometry import sim3,transform
from recompose.original_gaussians import deleted_indices,movement_transform
from recompose.scene import export_scene


def gaussian_file(path):
    p=np.array([[0.,0,2],[0,0,2],[1,1,2],[1,1,2]])
    sh=np.arange(4*16*3,dtype=float).reshape(4,16,3)/1000
    sh[3]=sh[2]  # exact duplicate rows must retain their multiplicity
    write_gaussians(path,p,np.full((4,3),.1),np.tile([1.,0,0,0],(4,1)),np.full(4,.9),sh)
    return ply(path)['vertex'].data


def test_exact_subtraction_distinguishes_attributes_and_duplicate_multiplicity(tmp_path):
    original=gaussian_file(tmp_path/'original.ply')
    # Same XYZ does not mean the same Gaussian; SH distinguishes rows 0 and 1.
    retained=original[[1,2]]
    np.testing.assert_array_equal(deleted_indices(original,retained),[0,3])
    changed=retained.copy();changed['opacity'][0]+=.001
    with pytest.raises(ValueError,match='exact unchanged subset'):deleted_indices(original,changed)
    with pytest.raises(ValueError,match='more copies'):deleted_indices(original,original[[2,2,2]])


def test_original_world_gaussians_and_generated_mesh_share_one_movement(tmp_path):
    from recompose.render import load_gaussian_arrays
    original=tmp_path/'original.ply';before=gaussian_file(original).copy()
    vertices=np.array([[0.,0,0],[.3,0,0],[0,.3,0]])
    asset=tmp_path/'asset.npz';np.savez(asset,vertices=vertices,faces=np.array([[0,1,2]]),colors=np.ones((3,3))*.5)
    bg=tmp_path/'bg.ply';save_mesh(bg,vertices,np.array([[0,1,2]]))
    reference=sim3(1.7,Rotation.from_euler('xyz',[.2,.4,-.3]).as_matrix(),[3,1,2])
    node=dict(id='1',mesh=fingerprint(asset),gaussians=fingerprint(original),gaussian_coordinates='original_world',
              original_pose=reference.tolist(),object_to_world=reference.tolist())
    scene=dict(schema_version=2,status='complete',coordinate_convention='COLMAP world',background=dict(mesh=fingerprint(bg),gaussians=fingerprint(original)),objects=[node])
    write_json(tmp_path/'scene.json',scene);export_scene(tmp_path/'scene.json',tmp_path/'initial')
    exported=ply(tmp_path/'initial/1_gaussians.ply')['vertex'].data
    assert exported.tobytes()==before.tobytes()
    np.testing.assert_array_equal(movement_transform(node),np.eye(4))
    delta=sim3(1.,Rotation.from_euler('z',.3).as_matrix(),[.2,-.1,0])
    node['object_to_world']=(delta@reference).tolist()
    write_json(tmp_path/'moved.json',scene);export_scene(tmp_path/'moved.json',tmp_path/'moved')
    moved=load_gaussian_arrays(tmp_path/'moved/1_gaussians.ply')[0]
    expected=transform(np.c_[before['x'],before['y'],before['z']],delta)
    np.testing.assert_allclose(moved,expected,atol=1e-6)
    mesh=ply(tmp_path/'moved/1_mesh.ply')['vertex'].data
    np.testing.assert_allclose(np.c_[mesh['x'],mesh['y'],mesh['z']],transform(transform(vertices,reference),delta),atol=1e-6)
    assert ply(original)['vertex'].data.tobytes()==before.tobytes()
