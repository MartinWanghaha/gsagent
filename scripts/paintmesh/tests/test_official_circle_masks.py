"""Native square dilation parity and fully occluded camera frames."""
import ast
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/paintmesh'))
sys.path.insert(0, str(ROOT/'submodules/Inpaint360GS/tools'))
import paintmesh_normal as normals
from track_holes_headless import hole_reference, correct_prediction


def isolated_function(path, name, namespace):
    # Avoid loading unrelated CUDA extensions or LaMa's pinned Lightning stack.
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


clean_mask = isolated_function(ROOT/'submodules/Inpaint360GS/tools/prepare_paintmesh_lama_data.py',
    '_clean_mask', dict(np=np, cv2=cv2, LamaDataError=ValueError))


def test_square_matches_pil_maxfilter_and_legacy_is_unchanged():
    raw = np.zeros((101,121),bool)
    raw[:10,:12] = True
    raw[40:60,40:60] = True
    raw[43:57,43:57] = False
    reference = np.array(Image.fromarray(raw.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(21))) > 0
    np.testing.assert_array_equal(clean_mask(raw,50,10,'square'), reference)
    legacy = cv2.dilate(raw.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))) != 0
    np.testing.assert_array_equal(clean_mask(raw,50,10),legacy)
    assert not np.array_equal(reference,legacy)


def test_tiny_equal_components_choose_first_like_official():
    mask=np.zeros((100,100),bool);mask[10,10]=True;mask[80,80]=True
    result=clean_mask(mask,50,0,'square')
    assert result.sum()==1 and result[10,10]


def test_occluded_frame_is_empty_and_never_propagates_a_hallucinated_hole():
    image=np.zeros((100,100,3),np.float32)
    mask=hole_reference(image,image,allow_empty=True)
    assert not mask.any()
    result,stats=correct_prediction(np.ones((100,100)),mask)
    assert not result.any() and stats['reset_reference']
    assert not clean_mask(mask,50,10,'square',True).any()
    with pytest.raises(ValueError):clean_mask(mask,50,10)
    with pytest.raises(ValueError):hole_reference(image,image)


def test_empty_normal_completion_preserves_every_value():
    source=np.zeros((8,9,3),np.float32);source[...,2]=-1
    valid=np.ones((8,9),bool);valid[0,0]=False;source[0,0]=0
    hole=np.zeros((8,9),bool)
    np.testing.assert_array_equal(normals.inference_mask(hole,valid),~valid)
    output,output_valid=normals.compose_normal(source,valid,hole,np.zeros_like(source),{})
    np.testing.assert_array_equal(output,source);np.testing.assert_array_equal(output_valid,valid)
    with pytest.raises(ValueError):normals.inference_mask(np.ones_like(hole),valid)


@pytest.mark.parametrize('foreground',[0.,1.])
def test_lama_refinement_empty_selections_have_finite_gradients(foreground):
    loss_fn=isolated_function(ROOT/'submodules/Inpaint360GS/LaMa/saicinpainting/evaluation/refinement.py',
        '_l1_loss',dict(torch=torch))
    pred=torch.full((1,3,8,8),.5,requires_grad=True)
    mask=torch.full_like(pred,foreground);reference=torch.zeros_like(pred)
    loss=loss_fn(pred,pred,reference,mask,mask,reference)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
