"""A downloaded config alone must never trigger GPU generation with partial weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wait_recompose import checkpoint_candidates


def test_queue_requires_every_complete_payload(tmp_path):
    requirements=dict(files=[dict(file='checkpoints/pipeline.yaml',size=5),dict(file='checkpoints/model.ckpt',size=12)])
    folder=tmp_path/'download/checkpoints';folder.mkdir(parents=True)
    assert checkpoint_candidates(tmp_path,requirements)==([],[])
    (folder/'pipeline.yaml').write_bytes(b'12345')
    (folder/'model.ckpt').write_bytes(b'partial')
    ready,pending=checkpoint_candidates(tmp_path,requirements)
    assert not ready and pending[0]['missing_or_partial']==['model.ckpt']
    (folder/'model.ckpt').write_bytes(b'123456789012')
    assert checkpoint_candidates(tmp_path,requirements)==([folder],[])


def test_queue_exposes_ambiguous_model_folders(tmp_path):
    requirements=dict(files=[dict(file='checkpoints/pipeline.yaml',size=5)])
    for name in ['first','second']:
        folder=tmp_path/name;folder.mkdir();(folder/'pipeline.yaml').write_bytes(b'12345')
    ready,pending=checkpoint_candidates(tmp_path,requirements)
    assert len(ready)==2 and not pending


def test_complete_modelscope_cache_is_accepted(tmp_path):
    folder=tmp_path/'.cache/modelscope/hub/models/facebook/sam-3d-objects/checkpoints'
    folder.mkdir(parents=True)
    (folder/'pipeline.yaml').write_bytes(b'12345')
    (folder/'model.ckpt').write_bytes(b'123456789012')
    (folder/'ss_encoder.safetensors').touch()
    requirements=dict(files=[dict(file='checkpoints/pipeline.yaml',size=5),
                            dict(file='checkpoints/model.ckpt',size=12),
                            dict(file='checkpoints/ss_encoder.safetensors',size=0)])
    assert checkpoint_candidates(tmp_path/'.cache',requirements)==([folder],[])
