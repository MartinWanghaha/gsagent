"""Anchor compression must preserve source geometry and distinct layers."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.depth_surface_anchors import build_frame_anchors,merge_anchors,PARAMETERS


def frame(height=32,width=32,curvature=0.):
    yy,xx=np.mgrid[:height,:width]
    K=np.array([[80.,0,(width-1)/2],[0,80.,(height-1)/2],[0,0,1.]])
    x=(xx-K[0,2])/K[0,0];y=(yy-K[1,2])/K[1,1]
    # rho is affine for an exact slanted 3-D plane.
    rho=1+.08*x-.05*y+curvature*(x*x+y*y)
    depth=(1/rho).astype('float32')
    # For q=r/rho(x,y), the normal is proportional to
    # [rho_x,rho_y,rho-x*rho_x-y*rho_y].
    rx=.08+2*curvature*x;ry=-.05+2*curvature*y
    normal=-np.stack([rx,ry,rho-x*rx-y*ry],-1)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    return dict(depth=depth,normal=normal.astype('float32'),K=K,w2c=np.eye(4),
        alpha=np.ones_like(depth),confidence=np.ones_like(depth)*.8,
        rgb=np.broadcast_to(np.array([60,120,180],dtype='uint8'),(height,width,3)).copy())


def test_slanted_plane_uses_exact_source_pixels_and_checks_every_native_pixel():
    source=frame();anchors,receipt=build_frame_anchors(source)
    assert receipt['all_native_pixels_checked'] and receipt['source_pixels_considered']==32*32
    assert len(anchors['positions'])<100
    assert receipt['maximum_accepted_axial_depth_error_pixels']<1e-4
    uv=anchors['pixels'];x,y=uv[:,0],uv[:,1]
    rays=np.c_[(x-source['K'][0,2])/80,(y-source['K'][1,2])/80,np.ones(len(x))]
    np.testing.assert_allclose(anchors['positions'],rays*source['depth'][y,x,None],atol=6e-8)
    np.testing.assert_array_equal(anchors['colors'],source['rgb'][y,x])
    assert not receipt['image_downsampling'] and receipt['no_point_cap']


def test_curved_patch_adaptively_adds_anchors_with_same_error_bound():
    planar,_=build_frame_anchors(frame(curvature=0))
    curved,receipt=build_frame_anchors(frame(curvature=4))
    assert len(curved['positions'])>len(planar['positions'])*3
    assert receipt['maximum_accepted_axial_depth_error_pixels']<=PARAMETERS['depth_error_pixels']+1e-8
    assert receipt['maximum_accepted_normal_cone_degrees']<=PARAMETERS['normal_cone_degrees']+1e-8


def test_depth_step_and_invalid_holes_are_not_interpolated_across():
    source=frame(19,23)
    source['depth'][:,11:]*=1.8
    source['depth'][8:11,7:9]=0
    anchors,receipt=build_frame_anchors(source)
    uv=anchors['pixels'];x,y=uv[:,0],uv[:,1]
    np.testing.assert_allclose(anchors['positions'][:,2],source['depth'][y,x],atol=1e-7)
    assert not np.any((y>=8)&(y<11)&(x>=7)&(x<9))
    assert receipt['source_pixels_considered']==19*23
    assert receipt['valid_source_pixels']==19*23-6
    assert np.any(x<11) and np.any(x>=11)


def test_nonidentity_camera_z_depth_and_unit_normal_rotation_are_correct():
    source=frame();angle=.6
    R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=[2.,-.3,1.4];source['w2c']=T
    anchors,_=build_frame_anchors(source)
    uv=anchors['pixels'];x,y=uv[:,0],uv[:,1]
    camera=anchors['positions'].astype('float64')@R.T+T[:3,3]
    np.testing.assert_allclose(camera[:,2],source['depth'][y,x],atol=2e-7)
    np.testing.assert_allclose(camera[:,:2]/camera[:,2,None]*80+source['K'][:2,2],uv,atol=2e-5)
    expected=source['normal'][y,x]@R
    np.testing.assert_allclose(anchors['normals'],expected,atol=6e-8)
    assert np.all(anchors['scale']>0)


def test_merge_preserves_separated_layers_and_opposite_orientations():
    original,_=build_frame_anchors(frame())
    # Duplicate observations with higher confidence retain their exact color;
    # points on another layer or with another orientation stay independent.
    duplicate={k:v.copy() for k,v in original.items()}
    duplicate['confidence'][:]=.9;duplicate['frame_indices'][:]=1;duplicate['colors'][:]=[10,20,30]
    other_layer={k:v.copy() for k,v in original.items()}
    other_layer['positions']+=other_layer['normals']*.02;other_layer['frame_indices'][:]=2
    opposite={k:v.copy() for k,v in original.items()}
    opposite['normals']*=-1;opposite['frame_indices'][:]=3
    merged,receipt=merge_anchors([original,duplicate,other_layer,opposite])
    assert len(merged['positions'])==3*len(original['positions'])
    assert receipt['merged_observations']==len(original['positions'])
    best=merged['frame_indices']==1
    assert best.sum()==len(original['positions'])
    np.testing.assert_array_equal(merged['colors'][best],np.broadcast_to([10,20,30],(best.sum(),3)))
    assert not receipt['opposite_orientations_merged'] and not receipt['world_positions_averaged']
    swapped,_=merge_anchors([opposite,other_layer,duplicate,original])
    for key in merged:np.testing.assert_array_equal(merged[key],swapped[key])


def test_same_facing_subpixel_layer_outside_plane_tolerance_is_retained():
    first,_=build_frame_anchors(frame())
    second={k:v.copy() for k,v in first.items()}
    second['positions']+=second['normals']*(second['scale']*.35)[:,None]
    second['frame_indices'][:]=1
    merged,receipt=merge_anchors([first,second])
    assert len(merged['positions'])==2*len(first['positions'])
    assert receipt['merged_observations']==0


def test_single_row_and_all_invalid_are_accounted_without_downsampling():
    one,receipt=build_frame_anchors(frame(1,9))
    assert len(one['positions'])>0 and receipt['source_pixels_considered']==9
    invalid=frame(5,7);invalid['depth'][:]=0
    anchors,receipt=build_frame_anchors(invalid)
    assert len(anchors['positions'])==0 and receipt['source_pixels_considered']==35
    merged,record=merge_anchors([anchors])
    assert record['input_anchors']==record['output_anchors']==0
