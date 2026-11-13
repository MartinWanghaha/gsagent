"""Changing generated geometry must invalidate appearances and shared-domain scores."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sync_latest_whole_meshes import same_selection,invalidate_changed_outputs


def test_reference_selection_does_not_ignore_masks_or_targets():
    original=dict(source='/data',mask_root='/masks',target_ids=[18],background_mesh='old.ply')
    assert same_selection(original,dict(original,background_mesh='whole.ply'))
    assert not same_selection(original,dict(original,target_ids=[18,19]))
    assert not same_selection(original,dict(original,mask_root='/other'))


def test_changed_mesh_cannot_keep_stale_color_field_or_renders(tmp_path):
    changed=tmp_path/'recomposition/paintmesh/bag/renders/view.png'
    retained=tmp_path/'background/paintmesh/bag/renders/view.png'
    baseline=tmp_path/'recomposition/gof/bag/renders/view.png'
    paths=[changed,retained,baseline,tmp_path/'recomposition/gof/bag/metrics.json',
           tmp_path/'common/bag/recomposition_support/complete.json',
           tmp_path/'cache/paintmesh/bag/recomposition/complete.json',
           tmp_path/'cache/paintmesh/bag/background/complete.json']
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('previous')
    invalidate_changed_outputs(tmp_path,'bag')
    assert not changed.exists()
    assert retained.read_text()==baseline.read_text()=='previous'
    assert not (tmp_path/'recomposition/gof/bag/metrics.json').exists()
    assert not (tmp_path/'common/bag/recomposition_support').exists()
    assert not (tmp_path/'cache/paintmesh/bag/recomposition').exists()
    assert (tmp_path/'cache/paintmesh/bag/background/complete.json').exists()
