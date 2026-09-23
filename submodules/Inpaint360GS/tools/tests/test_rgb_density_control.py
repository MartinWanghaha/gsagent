from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from edit_object_inpaint import update_inpaint_density


@pytest.mark.parametrize('iteration', [0, 500, 600, 4900, 5000])
def test_disabled_never_accesses_statistics_or_topology(iteration):
    # No Gaussian attributes: any accidental statistics access fails the test.
    update_inpaint_density(False, iteration, object(), None, None, None, None, None)


@pytest.mark.parametrize('iteration,stats,densify', [
    (0,1,0),(500,1,0),(501,1,0),(600,1,1),(4900,1,1),(5000,0,0)])
def test_enabled_preserves_original_schedule(iteration, stats, densify):
    model=SimpleNamespace(max_radii2D=torch.zeros(2),sub_feature_num=1,
        add_densification_stats=Mock(),densify_and_prune_inpaint=Mock())
    opt=SimpleNamespace(densify_grad_threshold=.0002)
    update_inpaint_density(True,iteration,model,None,torch.tensor([True,False]),
                           torch.tensor([3.,4.]),opt,2.)
    assert model.add_densification_stats.call_count==stats
    assert model.densify_and_prune_inpaint.call_count==densify
    assert model.max_radii2D.tolist()==([3.,0.] if stats else [0.,0.])
    if densify:
        model.densify_and_prune_inpaint.assert_called_once_with(.0002,.005,2.,20,1)
