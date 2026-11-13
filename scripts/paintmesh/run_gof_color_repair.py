#!/usr/bin/env python3
"""Two-GPU GOF color-range repair, native rerender, scoring and publication."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_benchmark.core import BASE, SCENES, STATES, read, write
from mesh_benchmark.layout import work_root
from recompose.io import fingerprint
from repair_gof_mesh_colors import PUBLIC, WORK
from evaluate_native_mesh_rgb import gpus, available_ram

PYTHON = Path('/home/member/wyc/mesh_gs_research/envs/come_probe/bin/python')
MESH_PYTHON = Path('/home/member/wyc/mesh_gs_research/envs/meshgs_wrapping/bin/python')


def snapshot():
    root = work_root(PUBLIC)
    before = WORK / 'before'
    if (before / 'complete.json').exists():
        return
    for scene in SCENES:
        dest = before / scene
        dest.mkdir(parents=True, exist_ok=True)
        src = root / 'native_renderers/gof' / scene
        shutil.copytree(src / 'renders', dest / 'renders', dirs_exist_ok=True)
        write(dest / 'native_complete.json', read(src / 'complete.json'))
        depths = {p.name: fingerprint(p)['sha256'] for p in (src / 'depth').glob('*.npy')}
        write(dest / 'depth_hashes.json', depths)
        for state in STATES:
            shutil.copy2(root / state / 'gof' / scene / 'metrics.json', dest / (state + '_metrics.json'))
            shutil.copytree(root / 'common' / scene / (state + '_native_support'), dest / (state + '_native_support'), dirs_exist_ok=True)
    write(before / 'complete.json', dict(complete=True, time=time.time(), purpose='Immediately before GOF color-range correction'))


def status(stage, tasks):
    write(WORK / 'status.json', dict(complete=stage == 'complete', stage=stage, tasks=tasks, updated=time.time(), pid=os.getpid()))
    lines = ['# GOF 彩色斑块修复进度', '', f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。', '',
             '已用原生查询验证 uint8 溢出；仅修正超范围颜色通道，所有坐标、面索引及范围内颜色字节保留。完成后自动覆盖原生渲染、两个报告和全部子图。', '',
             '| 场景 | 状态 | GPU |', '|---|---|---|']
    lines += [f'| {s} | {r["status"]} | {r.get("gpu", "—")} |' for s, r in tasks.items()]
    (PUBLIC / 'docs/GOF_COLOR_REPAIR_STATUS.md').write_text('\n'.join(lines) + '\n')


def schedule(tasks, render=False):
    sys.path.insert(0, str(BASE / '_scripts'))
    from common import environment
    active = {}
    stage = 'native render' if render else 'native float RGB and saturation'
    while True:
        for gpu, (scene, proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            tasks[scene].update(status='complete' if code == 0 else 'failed', exit_code=code)
            del active[gpu]
            print('GOF_COLOR_TASK_END', stage, scene, code, flush=True)
        for gpu, free in gpus():
            if gpu in active or free < 12000 or available_ram() < 24:
                continue
            pending = [s for s, row in tasks.items() if row['status'] == 'pending']
            if not pending:
                break
            scene = pending[0]
            if render:
                command = [str(MESH_PYTHON), str(HERE / 'render_native_colored_meshes.py'), '--method', 'gof', '--scene', scene, '--output', str(PUBLIC)]
            else:
                command = [str(PYTHON), '-u', str(HERE / 'repair_gof_mesh_colors.py'), '--scene', scene, '--probe', '0']
            log = open(WORK / (scene + ('_render.log' if render else '_repair.log')), 'w')
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment(gpu))
            tasks[scene].update(status='running', gpu=gpu, pid=proc.pid)
            active[gpu] = (scene, proc, log)
            print('GOF_COLOR_TASK_START', stage, scene, gpu, proc.pid, flush=True)
        status(stage, tasks)
        if not active and not any(r['status'] == 'pending' for r in tasks.values()):
            break
        time.sleep(10)
    if any(r['status'] != 'complete' for r in tasks.values()):
        raise RuntimeError('GOF color repair failed; inspect per-scene logs and status')


def publish():
    root = work_root(PUBLIC)
    figures = PUBLIC / 'figures_all_methods/gof_color_repair'
    figures.mkdir(exist_ok=True)
    selection = read(PUBLIC / 'figures_all_methods/metadata/selected_views.json')['scenes']
    records = []
    for scene in SCENES:
        receipt = read(WORK / scene / 'complete.json')
        rows = []
        for state in STATES:
            old = read(WORK / 'before' / scene / (state + '_metrics.json'))
            new = read(root / state / 'gof' / scene / 'metrics.json')
            assert old['geometry'] == new['geometry'], 'Color repair changed geometric evaluation'
            for item in old['per_view']:
                name = Path(item['name']).stem + '.png'
                np.testing.assert_array_equal(np.asarray(Image.open(WORK / 'before' / scene / (state + '_native_support') / name)),
                                              np.asarray(Image.open(root / 'common' / scene / (state + '_native_support') / name)))
            rows.append(dict(state=state, before=old['mean'], after=new['mean']))
        for name, sha in read(WORK / 'before' / scene / 'depth_hashes.json').items():
            assert fingerprint(root / 'native_renderers/gof' / scene / 'depth' / name)['sha256'] == sha, 'Color correction changed raster depth'
        records.append(dict(scene=scene, vertices=receipt['vertices'], changed_vertices=receipt['changed_vertices'],
                            changed_fraction=receipt['changed_vertices'] / receipt['vertices'], native_float_reproduction=receipt['original_float_reproduction'],
                            before_after=rows, receipt=str(WORK / scene / 'complete.json')))
        camera = selection[scene]['camera']
        name = camera['stem'] + '.png'
        paths = [WORK / 'before' / scene / 'renders' / name,
                 root / 'native_renderers/gof' / scene / 'renders' / name,
                 root / 'common' / scene / 'held_out' / (camera['stem'] + '_gt.png')]
        width = 650
        height = round(width * camera['height'] / camera['width'])
        canvas = Image.new('RGB', (width * 3, height + 40), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (path, label) in enumerate(zip(paths, ['Before: uint8 wrap', 'After: saturated native RGB', 'Reference RGB'])):
            canvas.paste(Image.open(path).convert('RGB').resize((width, height)), (i * width, 36))
            draw.text((i * width + 10, 10), scene + ' | ' + label, fill='black')
        canvas.save(figures / (scene + '.jpg'), quality=95)
    write(PUBLIC / 'docs/GOF_COLOR_REPAIR_RESULTS.json', dict(complete=True, updated=time.time(), records=records,
          geometry_metrics_exact=True, native_depth_bytes_exact=True, common_evaluation_masks_exact=True,
          in_range_color_bytes_exact=True, reference_data_used_for_repair=False))
    lines = ['# GOF 彩色斑块：颜色导出溢出已修复', '', f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。', '',
        '根因位于 GOF 原生 `extract_mesh.py:108`：`(color * 255).astype(np.uint8)` 未先截断范围。SH 积分颜色允许超过1；例如 `[1.05, 1.02, 0.95]` 被导出为 `[11, 4, 242]`，产生蓝紫色；正确饱和结果为 `[255, 255, 242]`。白色桌布、亮黄色物体最明显。', '',
        '先在原坐标重新调用原生 `evaluage_alpha`，使用 checkpoint 原有 filter_3D、原训练相机和 kernel_size。每个场景先抽查16,384点，随后逐批确认未经截断的 RGB 与原导出一致。修复仅将超出范围的通道饱和到0/255，其余已存颜色字节保持完全不变。未用 GT 着色、未平滑、未重训。', '',
        '所有顶点坐标、整数面索引、非RGB属性、原生深度文件、共同评分mask及CD/F1均逐项验证不变。以下是相同像素上的颜色修复结果；之前的跨空面几何修复另见 [几何修复记录](OPACITY_MESH_REPAIR.md)。', '',
        '| 场景 | 修正顶点比例 | 模式 | PSNR前 | 后 | SSIM前 | 后 | LPIPS前 | 后 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for row in r['before_after']:
            a, b = row['before'], row['after']
            lines.append(f'| {r["scene"]} | {r["changed_fraction"]:.3%} | {row["state"]} | {a["PSNR"]:.4f} | {b["PSNR"]:.4f} | {a["SSIM"]:.4f} | {b["SSIM"]:.4f} | {a["LPIPS"]:.4f} | {b["LPIPS"]:.4f} |')
    lines += ['', '[11场景前后对比图](../figures_all_methods/gof_color_repair/) · [完整记录](GOF_COLOR_REPAIR_RESULTS.json)', '',
              '修正溢出只消除非预期颜色错误。原生最小透射率选视角产生的颜色不连续、几何误差和高光仍属于GOF本身的剩余误差，不能保证达到高斯渲染的质量。', '',
              '[纯背景报告](../reports/background.md) · [组合mesh报告](../reports/recomposition.md) · [全方法图片](../figures_all_methods/README.md)']
    (PUBLIC / 'docs/GOF_COLOR_REPAIR.md').write_text('\n'.join(lines) + '\n')
    # Keep the earlier geometric before/after table as an accurately dated
    # historical experiment; do not attribute this color improvement to it.
    note = '**GOF 当前颜色另已修正导出溢出，最新主表和图片已更新。下表保留几何修复当时的前后结果；颜色修复单独见 [GOF颜色修复](GOF_COLOR_REPAIR.md)。**\n\n'
    p = PUBLIC / 'docs/OPACITY_MESH_REPAIR.md'
    text = p.read_text()
    if 'GOF 当前颜色另已修正导出溢出' not in text:
        p.write_text(note + text)
    for path in [PUBLIC / 'README.md', PUBLIC / 'docs/README.md', PUBLIC / 'docs/QUALITY_NOTES.md']:
        relative = 'docs/GOF_COLOR_REPAIR.md' if path.parent == PUBLIC else 'GOF_COLOR_REPAIR.md'
        text = path.read_text()
        line = f'\nGOF 全11场景颜色导出溢出已修复，渲染、指标和对比图已更新：[原因与同像素前后结果]({relative})。\n'
        if '颜色导出溢出已修复' not in text:
            path.write_text(text + line)
    return records


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = open(WORK / 'campaign.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot()
    priority = ['cube', 'fruits'] + [s for s in SCENES if s not in ['cube', 'fruits']]
    tasks = {s: dict(status='pending') for s in priority}
    schedule(tasks)
    tasks = {s: dict(status='pending') for s in priority}
    schedule(tasks, render=True)
    root = work_root(PUBLIC)
    protocol = read(root / 'protocol.json')
    protocol['native_color_range_correction'] = dict(enabled=True, method='gof', scenes=SCENES,
        rule='Saturate only out-of-range native RGB channels; retain in-range color bytes', geometry_unchanged=True)
    write(root / 'protocol.json', protocol)
    finish(tasks)


def finish(tasks=None):
    if tasks is None:tasks={s:dict(status='complete') for s in SCENES}
    status('scoring and comparison figures', tasks)
    with (WORK / 'evaluate.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(HERE / 'evaluate_native_mesh_rgb.py'), '--output', str(PUBLIC)], stdout=log, stderr=subprocess.STDOUT, check=True)
    publish()
    from verify_native_mesh_delivery import verify
    verify(PUBLIC)
    status('complete', tasks)
    print('GOF_COLOR_REPAIR_DELIVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--finish-only',action='store_true');a=p.parse_args()
    if a.finish_only:
        lock=open(WORK/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finish()
    else:run()
