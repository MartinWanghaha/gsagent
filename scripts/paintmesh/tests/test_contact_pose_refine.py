"""Rigid support correction must preserve geometry and report unresolved cases."""
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.contact_pose_refine import refine_contact_pose,refine_contact_pose_envelope,rigid_delta


def plane(slope=0.):
    v=np.array([[-2,-2,-2*slope],[2,-2,2*slope],[2,2,2*slope],[-2,2,-2*slope]],float)
    return v,np.array([[0,1,2],[0,2,3]],dtype='int32')


def box():
    m=trimesh.creation.box(extents=[1,1,1])
    return np.array(m.vertices),np.array(m.faces)


def test_box_resolves_actual_plane_intersection_preserving_mesh_and_scale():
    v,f=box();ev,ef=plane();original_v=v.copy();original_f=f.copy()
    T=np.eye(4);T[2,3]=.4
    result,report=refine_contact_pose(v,f,T,ev,ef,up=[0,0,1],config={
        'max_translation_fraction':.15,'step_translation_fraction':.1,'verify_intersections':True})
    assert report['before']['maximum_penetration_fraction']>.05
    assert report['after']['maximum_penetration_fraction']<1e-6
    assert report['intersection_counts']['transverse']==0
    assert 0<result[2,3]-.5<.001
    assert report['scale_preserved'] and np.array_equal(v,original_v) and np.array_equal(f,original_f)


def test_nonhorizontal_triangle_support_can_change_rotation_without_scale():
    v,f=box();ev,ef=plane(.2);T=np.eye(4);T[2,3]=.48
    result,report=refine_contact_pose(v,f,T,ev,ef,up=[0,0,1],config={
        'max_translation_fraction':.15,'max_rotation_radians':.25,
        'step_translation_fraction':.1,'step_rotation_radians':.15,'verify_intersections':True})
    assert report['after']['maximum_penetration_fraction']<1e-6
    assert report['intersection_counts']['transverse']==0
    assert report['rotation_radians']>1e-4
    assert report['scale_preserved']


def test_infeasible_small_trust_region_is_not_certified_collision_free():
    v,f=box();ev,ef=plane();T=np.eye(4);T[2,3]=.3
    result,report=refine_contact_pose(v,f,T,ev,ef,up=[0,0,1],config={
        'max_translation_fraction':.005,'max_rotation_radians':.001,'verify_intersections':True})
    assert report['translation_fraction']<=.005+1e-8
    assert report['rotation_radians']<=.001+1e-8
    assert report['after']['maximum_penetration_fraction']>0
    assert not report['collision_free_certified']
    assert report['intersection_counts']['transverse']>0


def test_rigid_delta_rotates_about_world_centroid_without_consuming_translation():
    T=np.eye(4);T[:3,3]=[100,-30,2]
    d=np.r_[.08,0,0,0,0,0]
    result=rigid_delta(T,d,T[:3,3],2)
    np.testing.assert_allclose(result[:3,3],T[:3,3],atol=1e-12)
    np.testing.assert_allclose(result[:3,:3].T@result[:3,:3],np.eye(3),atol=1e-12)


def test_envelope_uses_actual_nonplanar_triangle_heights_and_returns_pareto_candidates():
    v,f=box()
    # Four triangles form a piecewise-linear support ridge, not one plane.
    ev=np.array([[-2,-2,0],[0,-2,.1],[2,-2,0],[-2,2,0],[0,2,.1],[2,2,0]],float)
    ef=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]],dtype='int32')
    T=np.eye(4);T[2,3]=.52
    result,report=refine_contact_pose_envelope(v,f,T,ev,ef,up=[0,0,1],config={
        'max_translation_fraction':.1,'envelope_max_guard_candidates':4})
    assert report['before']['maximum_penetration_fraction']>0
    assert report['after']['maximum_penetration_fraction']<1e-6
    assert len(report['accepted_candidates'])==4
    assert report['translation_fraction']<=.1+1e-8
    assert report['scale_preserved']
    assert not report['collision_free_certified']
    assert report['complete_mesh_collision_recheck_required']
