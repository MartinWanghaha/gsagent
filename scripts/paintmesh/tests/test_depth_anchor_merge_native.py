from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.depth_surface_anchors import merge_anchors as reference,empty_anchors
from mesh_benchmark.depth_anchor_merge_native import merge_anchors as native


def arrays(points,normals=None,scale=None,confidence=None,frame=0):
    points=np.asarray(points,dtype='float32');count=len(points)
    n=np.tile([0,0,-1.],(count,1)) if normals is None else np.asarray(normals)
    n=n/np.linalg.norm(n,axis=1,keepdims=True)
    return dict(positions=points,normals=n.astype('float32'),
        scale=np.full(count,.01,dtype='float32') if scale is None else np.asarray(scale,dtype='float32'),
        confidence=np.ones(count,dtype='float32') if confidence is None else np.asarray(confidence,dtype='float32'),
        colors=np.column_stack([np.arange(count)%255,np.full(count,40),np.full(count,90)]).astype('uint8'),
        pixels=np.column_stack([np.arange(count)%100,np.arange(count)//100]).astype('int32'),
        frame_indices=np.full(count,frame,dtype='int32'),anchor_spacing=np.full(count,.02,dtype='float32'))


def compare(inputs):
    expected,er=reference(inputs);actual,ar=native(inputs)
    assert actual.keys()==expected.keys()
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
    for key in er:
        if key.startswith('maximum_merged_'):
            np.testing.assert_allclose(ar[key],er[key],atol=2e-7,rtol=2e-12,err_msg=key)
        else:assert ar[key]==er[key],(key,ar[key],er[key])
    return actual,ar


def test_thousand_mixed_scale_random_observations_match_all_arrays_and_counts():
    rng=np.random.default_rng(7843);base=rng.normal(size=(250,3))*.02
    base[:,2]=1.+rng.normal(size=250)*.0001
    normals=np.c_[rng.normal(size=(250,2))*.01,-np.ones(250)]
    scales=.004*2.**rng.integers(0,4,250);outputs=[]
    for frame in range(4):
        points=base+rng.normal(size=base.shape)*(scales[:,None]*.1)
        outputs.append(arrays(points,normals,scales,rng.uniform(.2,1.,250),frame))
    compare(outputs)


def test_thousand_overlapping_thin_layers_and_opposite_normals_remain_separate():
    rng=np.random.default_rng(989);xy=rng.uniform(-.03,.03,size=(250,2));outputs=[]
    for frame,(z,sign) in enumerate([(1.,-1),(1.,1),(1.01,-1),(1.01,1)]):
        points=np.c_[xy,np.full(250,z)];normals=np.tile([0,0,sign],(250,1))
        outputs.append(arrays(points,normals,confidence=np.full(250,1-frame*.1),frame=frame))
    actual,_=compare(outputs)
    assert set(np.sign(actual['normals'][:,2]))=={-1.,1.}
    assert len(np.unique(actual['positions'][:,2]))==2


def test_exact_distance_tie_and_source_frame_order_match_greedy_reference():
    inputs=[arrays([[-.004,0,1],[.004,0,1],[0,0,1]],confidence=[1,.9,.1])]
    actual,_=compare(inputs)
    np.testing.assert_array_equal(actual['observation_counts'],[2,1])
    first=arrays([[0,0,1]],confidence=[.8],frame=1)
    second=arrays([[.001,0,1]],confidence=[.8],frame=0)
    a,_=compare([first,second]);b,_=compare([second,first])
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_hash_boundaries_and_exact_scale_ratio_preserve_source_decisions():
    points=[];scales=[];confidence=[]
    for axis in range(3):
        for side in [-1.,1.]:
            center=np.array([-.02,.01,1.]);center[axis]=side*.016
            for step,scale in [(0,.008),(.0039,.008),(.0041,.008),(.001,.016),(.001,.032)]:
                point=center.copy();point[axis]+=step
                points.append(point);scales.append(scale);confidence.append(1./(1+len(points)))
    compare([arrays(points,scale=scales,confidence=confidence)])


def test_empty_reference_schema_unchanged():
    actual,ar=native([empty_anchors()]);expected,er=reference([empty_anchors()])
    assert ar==er and actual.keys()==expected.keys()
    for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
