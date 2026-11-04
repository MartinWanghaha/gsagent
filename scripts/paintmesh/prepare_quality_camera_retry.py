#!/usr/bin/env python3
"""Re-render an isolated corrected model using an unchanged verified trajectory."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from run_geometry_ablation import read, write, now, sha
from virtual_render_io import read_camera_manifest, validate_render

REPO = Path(__file__).resolve().parents[2]


def run(case):
    root = Path(case['removal_root']); source = Path(case['source_removal'])
    archive = root / 'rejected_camera_attempt'
    if not (archive / 'archived.json').exists():
        # Only this repair's unconsumed virtual output is moved; all original
        # baseline files, corrected geometry and training products stay intact.
        archive.mkdir(exist_ok=True)
        for name in ('tracker', 'work_model/virtual'):
            old = root / name; dest = archive / Path(name).name
            if old.exists():
                if dest.exists(): raise ValueError('Ambiguous retry archive')
                shutil.move(str(old), str(dest))
        write(archive / 'archived.json', dict(time=now(), reason='Replanned cameras produced unobservable removal footprint; no training consumed these frames.'))
    cameras = root / 'tracker/virtual_cameras.json'; cameras.parent.mkdir(exist_ok=True)
    src = source / 'tracker/virtual_cameras.json'
    if cameras.exists(): assert sha(cameras) == sha(src)
    else: shutil.copy2(src, cameras)
    camera_data = read_camera_manifest(cameras)
    assert camera_data['frame_count'] == 180
    pair = root / 'work_model/virtual/virtual_render_manifest.json'
    if not pair.exists() or not read(pair).get('complete'):
        subprocess.run([sys.executable, str(REPO/'scripts/paintmesh/render_virtual_views.py'),
            '--backend', 'edgs-pgsr', '--model-path', str(root/'work_model'),
            '--edgs-model-path', str(REPO/'output/paintmesh/Inpaint360GS/inpaint360'/case['scene']/'edgs'),
            '--source-path', '/home/member/wyc/data/Inpaint360GS/inpaint360/'+case['scene'],
            '--iteration', '2000', '--resolution', '4', '--images', 'images', '--alpha-min', '.01',
            '--camera-manifest', str(cameras), '--tracker-archive', str(root/'tracker/images.zip')], check=True)
    render_path = root / 'work_model/virtual/ours_object_removal/iteration_2000/render_manifest.json'
    rendered = validate_render(render_path.parent)
    def file_record(p):
        stat = p.stat()
        return dict(path=str(p.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(p))
    session = root/'tracker/tracking_session.json'
    if not session.exists():
        write(session, dict(schema_version=2, kind='paintmesh-tracking-session', complete=False,
            status='in_progress', started_at=now(), started_at_ns=time.time_ns(),
            input_archive=file_record(root/'tracker/images.zip'),
            input_cameras=dict(file_record(cameras), artifact_id=camera_data['artifact_id']),
            expected_masks=[c['image_name']+'.png' for c in camera_data['cameras']],
            input_virtual_render=dict(path=str(render_path.resolve()), sha256=sha(render_path),
                artifact_id=rendered['artifact_id'], backend=rendered['backend'])))
    write(root/'tracker/camera_reuse.json',dict(complete=True,source=file_record(src),output=file_record(cameras),
        reason='Hold the original 180 adaptive cameras fixed while isolating target membership. Original trajectory provenance is retained unchanged.',
        no_test_data_used=True,corrected_target_selection=file_record(root/'target_selection.json')))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--scene',required=True)
    a=p.parse_args();run(next(c for c in read(a.plan) if c['scene']==a.scene))
