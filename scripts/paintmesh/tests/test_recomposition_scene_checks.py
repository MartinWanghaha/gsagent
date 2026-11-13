"""Delivered files must survive full geometry and hierarchy audits, including corruption."""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_recomposition_scenes import mesh_arrays,check_gltf,check_world_glb,fresh_placement_passed
from recompose.io import save_mesh,fingerprint,read_json
from recompose.mesh_export import export_mesh_scene
from recompose.texture import transform_glb


def test_export_audit_detects_same_size_background_corruption(tmp_path):
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
    source=tmp_path/'bg.ply';save_mesh(source,v,f,np.full((3,3),.5))
    scene=dict(status='complete',background=dict(mesh=fingerprint(source)),objects=[])
    path=export_mesh_scene(scene,tmp_path/'world')
    assert check_gltf(path,scene)['background_full_geometry']
    buffer=path.parent/read_json(path)['buffers'][0]['uri']
    with buffer.open('r+b') as h:h.write(struct.pack('<f',.123))
    with pytest.raises(AssertionError):check_gltf(path,scene)


def test_corrupt_triangle_indices_are_rejected():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    with pytest.raises(ValueError,match='outside vertex'):
        mesh_arrays(v,np.array([[0,1,3]]),np.ones((3,3)))


def test_world_glb_preserves_payload_and_requires_correct_pose(tmp_path):
    document=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=[0])],nodes=[dict(name='native')])
    payload=json.dumps(document).encode();payload+=b' '*(-len(payload)%4)
    content=struct.pack('<II',len(payload),0x4E4F534A)+payload+struct.pack('<II',4,0x004E4942)+b'1234'
    source=tmp_path/'source.glb';source.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(content))+content)
    T=np.eye(4);T[:3,3]=[1.,2,3];dest=tmp_path/'world.glb';transform_glb(source,dest,T)
    assert check_world_glb(source,dest,T)['native_payload_exact']
    with pytest.raises(ValueError,match='placement differs'):check_world_glb(source,dest,np.eye(4))


def test_fresh_constraints_override_published_complete_status():
    cfg=dict(minimum_iou=.75,minimum_up_confidence=.6,maximum_contact_gap=.015,maximum_penetration=.02)
    physics=dict(contact_gap_fraction=.01,oriented_penetration_max=.001,unweighted_penetration_max=.002,volumetric_penetration_max=0.)
    geometry=dict(validation_iou_median=.9,up_confidence=.9,physics=physics)
    report=dict(status='complete',integrity_passed=True,objects=[dict(config=cfg,fresh_geometry=geometry)],
                checks=[dict(name='fresh_scene_collision_checks',passed=True,details=dict(accepted=True))])
    assert fresh_placement_passed(report)
    physics['contact_gap_fraction']=.02
    assert not fresh_placement_passed(report)
    physics['contact_gap_fraction']=float('nan')
    assert not fresh_placement_passed(report)
