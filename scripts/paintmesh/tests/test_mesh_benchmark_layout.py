import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.layout import MARKER,work_root,public_root,figure_bucket,document_links,publish_reports


def test_legacy_output_keeps_flat_paths(tmp_path):
    figures=tmp_path/'figures_all_methods';figures.mkdir()
    assert work_root(tmp_path)==tmp_path
    assert public_root(tmp_path)==tmp_path
    assert figure_bucket(figures,'overview')==figures


def test_public_and_work_view_resolve_to_same_completed_data(tmp_path):
    work=tmp_path/'_internal/workspace';work.mkdir(parents=True)
    cfg={'public_root':str(tmp_path),'work_root':str(work)}
    (tmp_path/MARKER).write_text(json.dumps(cfg));(work/MARKER).symlink_to(tmp_path/MARKER)
    data=tmp_path/'results/background';data.mkdir(parents=True)
    (data/'metrics.json').write_text('{"PSNR": 25.0}')
    (work/'background').symlink_to(data,target_is_directory=True)
    figures=tmp_path/'figures_all_methods';figures.mkdir()
    assert work_root(tmp_path)==work_root(work)==work
    assert public_root(work)==tmp_path
    assert (work/'background/metrics.json').read_bytes()==(data/'metrics.json').read_bytes()
    assert figure_bucket(figures,'overview')==figures/'overview'


def test_report_links_survive_symlink_parent_and_double_dot(tmp_path):
    work=tmp_path/'_internal/workspace';work.mkdir(parents=True)
    data=tmp_path/'results/background';data.mkdir(parents=True)
    (data/'figures').mkdir();(data/'figures/bag.jpg').write_bytes(b'image')
    docs=tmp_path/'docs';docs.mkdir();(docs/'PROTOCOL.md').write_text('protocol')
    (work/'background').symlink_to(data,target_is_directory=True)
    (work/'PROTOCOL.md').symlink_to(docs/'PROTOCOL.md')
    source=work/'background/report.md';dest=tmp_path/'reports/background.md'
    text='[image](figures/bag.jpg) [protocol](../PROTOCOL.md#mask) [web](https://example.com/a)'
    converted=document_links(text,source,dest)
    assert '[image](../results/background/figures/bag.jpg)' in converted
    assert '[protocol](../docs/PROTOCOL.md#mask)' in converted
    assert '[web](https://example.com/a)' in converted


def test_public_report_rebases_links_from_physical_result_directory(tmp_path):
    work=tmp_path/'_internal/workspace';work.mkdir(parents=True)
    cfg={'public_root':str(tmp_path),'work_root':str(work)}
    (work/MARKER).write_text(json.dumps(cfg))
    target=tmp_path/'docs/protocol.md';target.parent.mkdir();target.write_text('protocol')
    for state in ['background','recomposition']:
        data=tmp_path/'results'/state;data.mkdir(parents=True)
        (data/'report.md').write_text('[protocol](../../docs/protocol.md)')
        (work/state).symlink_to(data,target_is_directory=True)
    publish_reports(work)
    assert (tmp_path/'reports/background.md').read_text()=='[protocol](../docs/protocol.md)'
