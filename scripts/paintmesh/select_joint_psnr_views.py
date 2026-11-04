#!/usr/bin/env python3
"""Choose matched test views with jointly high full and official masked PSNR."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import make_paper_comparison as fig

OURS = 'PaintMesh 5000'
KEYS = ('PSNR', 'masked_PSNR')
PROTOCOL = (
    '使用 comparison.json 当前登记的 PaintMesh 5000 模型，逐场景对全部共同测试视角的 '
    'PSNR 和 masked_PSNR 分别计算升序百分位（越高越好，同分取平均排名）。'
    '以两项百分位的较小值作为主排序分数，以平均百分位为第二排序分数，再按文件名打破平局。'
    '每场景取前三个非空 mask 视角，总览取第一名；不按其他方法的分数或人工观感更换视角。'
    'masked_PSNR 使用官方定义：两幅图的 mask 外置零，再对全图像素计算 PSNR，'
    '不是只对 mask 内像素归一化的 PSNR；mask 面积也会影响该分数，逐帧面积随排名记录。'
    '四方法使用相同测试相机和裁剪；图中两项分数在各方法原生评估尺寸上计算，'
    '最多一像素的尺寸差异仅在制图时统一。属于择优展示，不代表全测试集表现；完整评估不变。'
)


def percentile(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError('Need at least three finite scores')
    return np.array([(np.count_nonzero(a < v) + .5 * (np.count_nonzero(a == v) - 1))
                     / (len(a) - 1) for v in a])


def native_scores(root, frame, mask_path):
    prediction = np.asarray(fig.rgb(root/'renders'/frame), dtype=np.float64) / 255
    gt = np.asarray(fig.rgb(root/'gt'/frame), dtype=np.float64) / 255
    if prediction.shape != gt.shape:
        raise ValueError(f'Render/GT size mismatch: {root}/{frame}')
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert('L').resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)) > 127
    if not mask.any():
        raise ValueError(f'Empty evaluation mask: {root}/{frame}')
    error = (prediction-gt)**2
    mses = (float(error.mean()), float((error*mask[..., None]).mean()))
    if any(v <= 0 or not np.isfinite(v) for v in mses):
        raise ValueError(f'Invalid MSE: {root}/{frame}')
    return dict(zip(KEYS, [float(-10*np.log10(v)) for v in mses]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=fig.DEFAULT_JOB)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    out = args.output_dir or job/'qualitative_pm5000_joint_psnr/selection'
    out.mkdir(parents=True, exist_ok=True)
    report = fig.read(job/'comparison.json')
    request = fig.read(job/'request.json')
    selection, ranked_all, selected_all = {}, [], []
    lines = ['# PSNR 与 masked_PSNR 联合择优选图', '', PROTOCOL, '',
             '各列为 **PSNR / masked_PSNR（dB）**；这里是实际渲染的逐视角值，不是论文表格数值。', '',
             '| 场景 | 联合排名 | 视角 | Inpaint360GS | Gaussian Grouping | AuraFusion360 | PaintMesh 5000 | 两项较低百分位 | mask 面积 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for scene in request['scenes']:
        name = scene['name']
        roots, frames, labels = fig.resolve_scene(job, scene, report)
        source = roots[OURS]/'results.json'
        result = fig.read(source)
        if not result['complete'] or result['inputs']['iteration'] != 5000:
            raise ValueError(f'Incomplete/wrong PaintMesh checkpoint: {source}')
        per_view = result['per_view']
        if set(per_view) != {Path(f).stem for f in frames}:
            raise ValueError('Incomplete per-view evaluation: '+name)
        scores = np.array([[per_view[Path(f).stem][k] for k in KEYS] for f in frames])
        for i, key in enumerate(KEYS):
            if abs(scores[:, i].mean()-report['metrics'][OURS][name][key]) > 1e-5:
                raise ValueError('Stale evaluation source: '+name+'/'+key)
        ranks = np.stack([percentile(scores[:, i]) for i in range(2)], axis=1)
        stats = {f: fig.mask_info(roots[OURS]/'masks'/f) for f in frames}
        rows = [dict(scene=name, frame=f, PSNR=float(scores[i, 0]), masked_PSNR=float(scores[i, 1]),
                     psnr_percentile=float(ranks[i, 0]), masked_psnr_percentile=float(ranks[i, 1]),
                     joint_score=float(ranks[i].min()), mean_percentile=float(ranks[i].mean()),
                     mask_fraction=stats[f]['area'] if stats[f] else 0)
                for i, f in enumerate(frames)]
        rows.sort(key=lambda r: (-r['joint_score'], -r['mean_percentile'], r['frame']))
        chosen = [r for r in rows if r['mask_fraction'] > 0][:3]
        if len(chosen) != 3:
            raise ValueError('Fewer than three visible repair views: '+name)
        frame_metrics, mask_records = {}, {}
        masks = {Path(m['path']).stem: m for m in result['inputs']['official_masks']}
        for rank, row in enumerate(rows, 1):
            ranked_all.append(dict(joint_rank=rank, selected=row in chosen, **row))
        for rank, row in enumerate(chosen, 1):
            frame = row['frame']
            mask = masks[Path(frame).stem]
            if fig.sha(mask['path']) != mask['sha256']:
                raise ValueError('Evaluation mask changed: '+mask['path'])
            fig.aligned_images(roots, frame)
            values = {method: native_scores(root, frame, Path(mask['path'])) for method, root in roots.items()}
            for key in KEYS:
                if abs(values[OURS][key]-row[key]) > 1e-4:
                    raise ValueError(f'Saved render and recorded score differ: {name}/{frame}/{key}')
                values[OURS][key] = row[key]
            frame_metrics[frame] = values
            mask_records[frame] = mask
            selected_all.append(dict(scene=name, joint_rank=rank, frame=frame,
                **{f'{m}/{k}': values[m][k] for m in fig.METHODS for k in KEYS},
                psnr_percentile=row['psnr_percentile'], masked_psnr_percentile=row['masked_psnr_percentile'],
                joint_score=row['joint_score'], mask_fraction=row['mask_fraction']))
            lines.append(f'| {name} | {rank} | {frame} | ' +
                ' | '.join(f'{values[m]["PSNR"]:.4f} / {values[m]["masked_PSNR"]:.4f}' for m in fig.METHODS) +
                f' | {row["joint_score"]:.1%} | {row["mask_fraction"]:.3%} |')
        selection[name] = dict(frames=[r['frame'] for r in chosen], source_test_frames=len(frames),
            scoring_source=str(source), scoring_source_sha256=fig.sha(source), per_frame_metrics=frame_metrics,
            selected_ranks=chosen, official_mask_sources=mask_records, correction_labels=labels)
        print(name, [(r['frame'], round(r['PSNR'], 2), round(r['masked_PSNR'], 2),
                      round(r['joint_score'], 3)) for r in chosen], flush=True)
    fig.save_json(out/'selected_frames.json', dict(created_at=datetime.now().astimezone().isoformat(),
        kind='joint_psnr', protocol=PROTOCOL, checkpoint=5000, report_source=str(job/'comparison.json'),
        report_sha256=fig.sha(job/'comparison.json'), scored_test_views=len(ranked_all), scenes=selection))
    for filename, records in [('all_views_ranked.csv', ranked_all), ('selected_view_metrics.csv', selected_all)]:
        with (out/filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    lines += ['', f'共排名 {len(ranked_all)} 个共同测试视角，选择 {len(selected_all)} 张。', '',
              '[全部视角排名](all_views_ranked.csv) · [入选四方法分数](selected_view_metrics.csv) · [机器可读选图记录](selected_frames.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
