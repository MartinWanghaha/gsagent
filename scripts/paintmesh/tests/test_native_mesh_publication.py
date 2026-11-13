"""Publication layout aliases must not create broken RGB links."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_native_mesh_rgb import publish_render_link


def test_render_link_through_organized_directory_alias(tmp_path):
    work=tmp_path/'internal/work';work.mkdir(parents=True)
    publication=tmp_path/'results/recomposition/come/bag';publication.mkdir(parents=True)
    (work/'recomposition').symlink_to(tmp_path/'results/recomposition',target_is_directory=True)
    source=work/'native_renderers/come/bag/renders/view.png'
    source.parent.mkdir(parents=True);source.write_bytes(b'native mesh RGB')
    destination=work/'recomposition/come/bag/renders/view.png'
    publish_render_link(source,destination)
    assert destination.read_bytes()==b'native mesh RGB'
    assert (publication/'renders/view.png').resolve()==source
    source.write_bytes(b'updated native mesh RGB')
    publish_render_link(source,destination)
    assert destination.read_bytes()==b'updated native mesh RGB'


def test_hash_reuse_preserves_full_validation_and_detects_restored_mtime(tmp_path,monkeypatch):
    import os
    import time
    import evaluate_native_mesh_rgb as evaluation
    source=tmp_path/'mesh.ply';source.write_bytes(b'original mesh bytes')
    actual=evaluation.fingerprint;calls=[]
    def measured(path):
        calls.append(path);return actual(path)
    monkeypatch.setattr(evaluation,'fingerprint',measured)
    # A stable model may reuse its digest. New writes must survive the kernel
    # ctime granularity window before their digests qualify for reuse.
    time.sleep(1.05)
    original=evaluation.verified_file_fingerprint(tmp_path/'work',source)
    assert original==actual(source)
    assert evaluation.verified_file_fingerprint(tmp_path/'work',source)==original
    assert len(calls)==1
    old=source.stat()
    source.write_bytes(b'modified mesh bytes')
    os.utime(source,ns=(old.st_atime_ns,old.st_mtime_ns))
    changed=evaluation.verified_file_fingerprint(tmp_path/'work',source)
    assert changed['sha256']!=original['sha256'] and len(calls)==2
    source.write_bytes(b'original mesh bytes')
    os.utime(source,ns=(old.st_atime_ns,old.st_mtime_ns))
    changed_again=evaluation.verified_file_fingerprint(tmp_path/'work',source)
    assert changed_again==original and len(calls)==3
    # A replacement inode also requires an entire SHA pass, even if bytes match.
    replacement=tmp_path/'replacement';replacement.write_bytes(source.read_bytes());replacement.replace(source)
    assert evaluation.verified_file_fingerprint(tmp_path/'work',source)==changed_again
    assert len(calls)==4
