#!/usr/bin/env python3
"""Select each scene's top three PaintMesh full-image PSNR test views."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import make_paper_comparison as fig

OURS = 'PaintMesh 5000'
PROTOCOL = (
    '按 PaintMesh 5000 原始评估 results.json 的逐帧全图 PSNR 从高到低排序，'
    '每场景严格取前三名共同测试视角，第一名用于总览。相同分数按文件名排序。'
    '不按 SSIM、masked PSNR、相对优势或人工观感调整排名，不增加视角间隔/面积筛选。'
    '四种方法使用同一测试相机和同一修复区域裁剪。图中标注的 PSNR 是每个方法在自己的原始评估尺寸上'
    '与该视角 GT 计算的全图 PSNR；仅制图时对部分相差 1 像素的尺寸做统一缩放。'
    '这是高 PSNR 择优展示，不代表全测试集平均表现；全图 PSNR 高也不保证洞内质量最佳。'
    '完整测试集评估结果保持不变。'
)


def native_psnr(root, frame):
    prediction = np.asarray(fig.rgb(root / 'renders' / frame), dtype=np.float64) / 255
    target = np.asarray(fig.rgb(root / 'gt' / frame), dtype=np.float64) / 255
    if prediction.shape != target.shape:
        raise ValueError(f'Render/GT dimensions differ: {root}/{frame}')
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 0:
        raise ValueError(f'Unexpected zero MSE: {root}/{frame}')
    return float(-10 * np.log10(mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=fig.DEFAULT_JOB)
    args = parser.parse_args()
    job = args.job.resolve()
    out = job / 'qualitative_pm5000_high_psnr/selection'
    out.mkdir(parents=True, exist_ok=True)
    request, report = fig.read(job / 'request.json'), fig.read(job / 'comparison.json')
    selection, ranking_rows, selected_rows = {}, [], []
    lines = ['# 按 PaintMesh 全图 PSNR 排名选取测试视角', '', PROTOCOL, '',
             '使用当前 comparison.md 登记的修复结果；星号对应修正版，AuraFusion360/redbull 为初始化修正版。', '',
             '| 场景 | 排名 | 帧号 | Inpaint360GS | Gaussian Grouping | AuraFusion360 | PaintMesh 5000 | mask 占图比例 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for scene in request['scenes']:
        name = scene['name']
        roots, frames, labels = fig.resolve_scene(job, scene, report)
        source = roots[OURS] / 'results.json'
        results = fig.read(source)
        per_view = results['per_view']
        if {Path(f).stem for f in frames} != set(per_view):
            raise ValueError(f'Per-view results do not cover the common test set: {name}')
        ranked = sorted(frames, key=lambda f: (-per_view[Path(f).stem]['PSNR'], f))
        mean = float(np.mean([per_view[Path(f).stem]['PSNR'] for f in frames]))
        if abs(mean - report['metrics'][OURS][name]['PSNR']) > 1e-5:
            raise ValueError(f'Registered mean does not match per-view results: {name}')
        frame_metrics = {}
        for rank, frame in enumerate(ranked, 1):
            stat = fig.mask_info(roots[OURS] / 'masks' / frame)
            value = per_view[Path(frame).stem]['PSNR']
            ranking_rows.append(dict(scene=name, rank=rank, frame=frame, PSNR=value,
                                     mask_fraction=stat['area'] if stat else 0))
            if rank > 3:
                continue
            if not stat:
                raise ValueError(f'Top-PSNR frame has no visible mask; explicitly handle it: {name}/{frame}')
            _, checks = fig.aligned_images(roots, frame)
            scores = {method: {'PSNR': native_psnr(root, frame)} for method, root in roots.items()}
            if abs(scores[OURS]['PSNR'] - value) > 1e-4:
                raise ValueError(f'Recorded PaintMesh PSNR disagrees with saved render: {name}/{frame}')
            scores[OURS]['PSNR'] = value
            frame_metrics[frame] = scores
            row = dict(scene=name, rank=rank, frame=frame, mask_fraction=stat['area'],
                       **{method: scores[method]['PSNR'] for method in fig.METHODS})
            selected_rows.append(row)
            lines.append(f'| {name} | {rank} | {frame} | ' +
                         ' | '.join(f"{scores[m]['PSNR']:.4f}" for m in fig.METHODS) +
                         f" | {100 * stat['area']:.3f}% |")
        selection[name] = dict(frames=ranked[:3], source_test_frames=len(frames),
                               scoring_source=str(source), scoring_source_sha256=fig.sha(source),
                               per_frame_metrics=frame_metrics, correction_labels=labels)
        print(name, [(f, round(frame_metrics[f][OURS]['PSNR'], 4)) for f in ranked[:3]], flush=True)
    fig.save_json(out / 'selected_frames.json', dict(
        created_at=datetime.now().astimezone().isoformat(), kind='full_psnr',
        protocol=PROTOCOL, checkpoint=5000, scored_test_views=len(ranking_rows), scenes=selection))
    for filename, rows in [('all_views_ranked.csv', ranking_rows), ('selected_view_psnr.csv', selected_rows)]:
        with (out / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines += ['', '[425 个测试视角的完整排名](all_views_ranked.csv) · [33 个展示视角的四方法 PSNR](selected_view_psnr.csv)', '',
              '复现：先运行 `scripts/paintmesh/select_high_psnr_views.py`，再使用生成的 `selected_frames.json` 调用 `make_paper_comparison.py --selection-json ... --output-dir ...`。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
