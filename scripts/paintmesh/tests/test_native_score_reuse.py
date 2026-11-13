from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_native_mesh_rgb as evaluation
from mesh_benchmark.core import read,write


@pytest.fixture
def reuse_case(tmp_path,monkeypatch):
    root=tmp_path;method='2dgs';scene='bag';state='background'
    common=root/'common'/scene;support=common/(state+'_native_support');support.mkdir(parents=True)
    (common/'held_out').mkdir();dest=root/state/method/scene;diagnostic=dest/'diagnostics';diagnostic.mkdir(parents=True)
    mask=np.array([[False,True],[True,True]])
    for path,value in [(common/'held_out/v_foreground.png',~mask),(support/'v.png',mask),
                       (diagnostic/'v_evaluation_mask.png',mask),(diagnostic/'v_coverage.png',mask)]:
        Image.fromarray(value.astype('uint8')*255).save(path)
    depth=root/'depth';depth.mkdir();np.save(depth/'v.npy',np.ones((2,2),dtype='float32'))
    geometry={'support_id':'fixed-geometry','F1':.8}
    write(root/'cache'/method/scene/state/'geometry_metrics.json',geometry)
    write(support/'complete.json',{'request_id':'new-cohort-identity'})
    request={'renderer':'same source mesh and renderer','support_id':'old-cohort-identity','geometry_support_id':'fixed-geometry'}
    old=dict(complete=True,native_mesh_rendering_version=1,request=request,geometry=geometry,mean={'PSNR':30.1234},per_view=[])
    write(dest/'metrics.json',old)
    source=dict(request='same source mesh and renderer',depth=depth)
    monkeypatch.setattr(evaluation,'cameras',lambda s:[{'part':'held_out','stem':'v'}])
    monkeypatch.setattr(evaluation,'rendering',lambda *args:source)
    return root,method,scene,state,source,old


def test_cohort_identity_only_change_reuses_scores_after_all_masks_match(reuse_case):
    root,method,scene,state,source,old=reuse_case
    assert evaluation.reuse_unchanged_score(root,method,scene,state)
    actual=read(root/state/method/scene/'metrics.json')
    assert actual['mean']==old['mean']
    assert actual['request']['support_id']=='new-cohort-identity'


def test_a_single_pixel_domain_change_or_color_source_change_forbids_reuse(reuse_case):
    root,method,scene,state,source,old=reuse_case
    source['request']='changed colors'
    assert not evaluation.reuse_unchanged_score(root,method,scene,state)
    source['request']='same source mesh and renderer'
    path=root/'common'/scene/(state+'_native_support')/'v.png'
    Image.fromarray(np.array([[0,0],[255,255]],dtype='uint8')).save(path)
    assert not evaluation.reuse_unchanged_score(root,method,scene,state)
