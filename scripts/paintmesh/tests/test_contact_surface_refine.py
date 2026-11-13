import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.contact_surface_refine import refine_contact_surface
from mesh_benchmark.mesh_intersection import audit_intersections
from mesh_benchmark.contact_heightfield_warp import refine_contact_surface_heightfield


def test_hidden_underside_obstacle_correction_preserves_topology_and_upper_shell():
    m=trimesh.creation.box(extents=[1,1,1]);v=np.array(m.vertices);f=np.array(m.faces)
    original=v.copy();T=np.eye(4);T[2,3]=.4
    ev=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],float)
    ef=np.array([[0,1,2],[0,2,3]],dtype='int32')
    out,report=refine_contact_surface(v,f,T,ev,ef,up=[0,0,1])
    assert report['geometry_modified']
    assert np.array_equal(v,original)
    np.testing.assert_array_equal(out[v[:,2]>.4],v[v[:,2]>.4])
    assert report['after']['penetrating_vertices']==0
    assert report['flipped_or_degenerate_faces']==0
    result=audit_intersections(out+T[:3,3],f,ev,ef)
    assert result['counts']['transverse']==0
    assert report['topology_unchanged'] and not report['collision_free_certified']


def test_piecewise_nonplanar_support_has_continuous_safe_contact_correction():
    m=trimesh.creation.box(extents=[1,1,.25]);m=m.subdivide()
    v=np.array(m.vertices);f=np.array(m.faces);T=np.eye(4);T[2,3]=.13
    ev=np.array([[-2,-2,0],[0,-2,.06],[2,-2,0],[-2,2,0],[0,2,.06],[2,2,0]],float)
    ef=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]],dtype='int32')
    out,report=refine_contact_surface(v,f,T,ev,ef,up=[0,0,1],config={'upper_pin_gap_fraction':.1})
    assert report['after']['maximum_penetration_fraction']<report['before']['maximum_penetration_fraction']
    assert report['flipped_or_degenerate_faces']==0
    assert report['visibility_pin_max_displacement']==0
    assert len(out)==len(v)


def test_unsafe_contact_patch_does_not_rollback_disconnected_safe_patch():
    m=trimesh.creation.box(extents=[.5,.5,.5]);a=np.array(m.vertices);f=np.array(m.faces)
    left=a+[-.7,0,0];right=a+[.7,0,0]
    v=np.vstack([left,right]);ff=np.vstack([f,f+len(a)])
    T=np.eye(4);T[2,3]=.2
    # A sloped left support produces a real normal change which this strict
    # test rejects; the independent right flat-bottom correction remains safe.
    ev=np.array([[-1.2,-.6,-.06],[-.2,-.6,.06],[-.2,.6,.06],[-1.2,.6,-.06],
                 [.2,-.6,0],[1.2,-.6,0],[1.2,.6,0],[.2,.6,0]],float)
    ef=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype='int32')
    out,report=refine_contact_surface(v,ff,T,ev,ef,up=[0,0,1],config={
        'minimum_face_normal_dot':.9999999,'local_safety_backoff_iterations':1,
        'local_safety_freeze_iterations':1})
    assert report['safety_frozen_vertices']>0
    assert np.any(np.linalg.norm(out[len(a):]-v[len(a):],axis=1)>1e-8)
    assert report['geometry_modified'] and report['flipped_or_degenerate_faces']==0


def test_continuous_heightfield_map_keeps_vertical_layer_order_and_resolves_plane():
    m=trimesh.creation.box(extents=[1,1,.2]);m=m.subdivide()
    v=np.array(m.vertices);f=np.array(m.faces);T=np.eye(4);T[2,3]=.05
    ev=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],float)
    ef=np.array([[0,1,2],[0,2,3]],dtype='int32')
    out,report=refine_contact_surface_heightfield(v,f,T,ev,ef,up=[0,0,1])
    assert report['after']['penetrating_vertices']==0
    assert report['theoretical_minimum_vertical_derivative']>0
    assert report['minimum_triangle_area_ratio']>0
    assert not report['local_freeze_used'] and not report['observation_pin_used']
    unique=np.unique(v[:,:2],axis=0)
    for xy in unique:
        ids=np.flatnonzero(np.all(v[:,:2]==xy,axis=1));ids=ids[np.argsort(v[ids,2])]
        assert np.all(np.diff(out[ids,2])>0)
    result=audit_intersections(out+T[:3,3],f,ev,ef)
    assert result['counts']['transverse']==0


def test_local_heightfield_correction_keeps_support_coverage_without_global_offset():
    m=trimesh.creation.box(extents=[1,1,.2]);m=m.subdivide()
    v=np.array(m.vertices);f=np.array(m.faces);T=np.eye(4);T[2,3]=.08
    ev=np.array([[-2,-2,0],[0,-2,.05],[2,-2,0],[-2,2,0],[0,2,.05],[2,2,0]],float)
    ef=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]],dtype='int32')
    out,report=refine_contact_surface_heightfield(v,f,T,ev,ef,up=[0,0,1],config={
        'heightfield_conservative_mode':'local','heightfield_smoothing_fraction':.01})
    assert report['conservative_height_correction_mode']=='local'
    assert report['support_envelope_residual_max']<1e-7
    assert report['after']['penetrating_vertices']==0
    assert len(report['local_correction_history'])<=10
