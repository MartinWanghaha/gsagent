from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_geometry_io as io


@pytest.mark.parametrize('enabled', [False, True])
def test_policy_reuse_and_switch_rejection(tmp_path, enabled):
    policy, ply = tmp_path/'policy.json', tmp_path/'points.ply'
    expected = io.prepare_rgb_policy(policy, ply, enabled)
    assert expected['parameters']['rgb_densify'] is enabled
    assert io.prepare_rgb_policy(policy, ply, enabled, validate_only=True) == expected
    with pytest.raises(ValueError, match='changed'):
        io.prepare_rgb_policy(policy, ply, not enabled)


def test_policy_refuses_unbound_results_and_missing_resume(tmp_path):
    policy, ply = tmp_path/'policy.json', tmp_path/'points.ply'
    with pytest.raises(ValueError, match='policy missing'):
        io.prepare_rgb_policy(policy, ply, validate_only=True)
    assert not policy.exists()
    ply.write_text('existing result')
    with pytest.raises(ValueError, match='policy missing'):
        io.prepare_rgb_policy(policy, ply)
    with pytest.raises(ValueError, match='boolean'):
        io.prepare_rgb_policy(policy, ply, enabled='false')


def test_rgb_context_binds_density_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'verify_targets', lambda *args: None)
    source = tmp_path/'input'; source.write_text('input')
    args = SimpleNamespace(**{k: source for k in (
        'source_ply','classifier','inpaint_config','camera','lama','fusion','support')},
        context=tmp_path/'context.json', manifest=tmp_path/'rgb.json',
        rgb_ply=tmp_path/'output.ply', rgb_iterations=5000, seed_frame=4,
        rgb_densify=False)
    io.prepare_rgb(args)
    assert io.read_json(args.context)['parameters']['rgb_densify'] is False
    args.rgb_densify=True
    with pytest.raises(ValueError, match='inputs changed'):
        io.prepare_rgb(args)
