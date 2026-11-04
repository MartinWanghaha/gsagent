#!/usr/bin/env python3
"""Select explicitly biased showcases: high own scores and large relative gaps."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import make_paper_comparison as fig
from select_joint_psnr_views import KEYS, OURS, native_scores, percentile

PROTOCOL = (
    '使用当前 comparison.json 登记的模型，重算全部共同测试视角的四方法 PSNR 与 masked_PSNR。'
    '先要求 PaintMesh 的两项分数分别不低于本场景中位数，保证自身两项表现都在较高区间。'
    '对每项计算 PaintMesh 分数减去三个其他方法中的最高分，得到相对最强对手的差值。'
    '在通过自身质量门槛的视角中，优先最大化两项差值中的较小值，再按平均差值、自身两项较低百分位、文件名排序。'
    '因此只有两项差值都为正，才能称为两项均高于全部其他方法；对手最高分可以来自不同方法。'
    '每场景取前三个共同视角，总览取第一名。若通过质量门槛的视角不足三张，'
    '剩余位置按自身两项较低百分位优先补足，并明确标注未同时达到中位数；不宣称这些补充视角满足门槛。'
    '无双指标全面领先视角的场景仍真实展示，不制造优势。该选集明确偏向 PaintMesh，不能代替完整测试集成绩。'
    '四方法使用相同相机、相同裁剪，不改动原始图像。图中分数在各方法原生评估尺寸上计算，'
    '最多一像素的尺寸差异仅在制图时统一。masked_PSNR 为 mask 外置零后对全图计算的官方式指标，'
    '不是洞内像素归一化 PSNR；逐帧 mask 面积单独记录。'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=fig.DEFAULT_JOB)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    out = args.output_dir or job/'qualitative_pm5000_relative_joint_psnr/selection'
    out.mkdir(parents=True, exist_ok=True)
    report = fig.read(job/'comparison.json')
    request = fig.read(job/'request.json')
    selections, all_csv, selected_csv, summaries = {}, [], [], []
    lines = ['# 自身双指标较高、相对其他方法优势较大的视角', '', PROTOCOL, '',
             '各方法单元格为 **PSNR / masked_PSNR（dB）**，差值相对该项最强对手；负值如实保留。', '',
             '| 场景 | 展示顺序 | 帧号 | Inpaint360GS | Gaussian Grouping | AuraFusion360 | PaintMesh 5000 | ΔPSNR | Δmasked_PSNR | 两项达到中位数 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for scene in request['scenes']:
        name = scene['name']
        roots, frames, labels = fig.resolve_scene(job, scene, report)
        source = roots[OURS]/'results.json'
        result = fig.read(source)
        assert result['complete'] and result['inputs']['iteration'] == 5000
        assert set(result['per_view']) == {Path(f).stem for f in frames}
        masks = {Path(r['path']).stem: r for r in result['inputs']['official_masks']}
        own_scores = np.array([[result['per_view'][Path(f).stem][k] for k in KEYS] for f in frames])
        ranks = np.stack([percentile(own_scores[:, i]) for i in range(2)], axis=1)
        medians = np.median(own_scores, axis=0)
        rows = []
        for i, frame in enumerate(frames):
            stat = fig.mask_info(roots[OURS]/'masks'/frame)
            if stat is None:
                raise ValueError(f'Empty evaluation mask: {name}/{frame}')
            mask = masks[Path(frame).stem]
            if fig.sha(mask['path']) != mask['sha256']:
                raise ValueError('Official evaluation mask changed: '+mask['path'])
            scores = {method: native_scores(root, frame, Path(mask['path'])) for method, root in roots.items()}
            for j, key in enumerate(KEYS):
                if abs(scores[OURS][key]-own_scores[i, j]) > 1e-4:
                    raise ValueError(f'Render/recorded score mismatch: {name}/{frame}/{key}')
                scores[OURS][key] = float(own_scores[i, j])
            strongest = {k: max((m for m in fig.METHODS if m != OURS), key=lambda m: scores[m][k]) for k in KEYS}
            gaps = {k: scores[OURS][k]-scores[strongest[k]][k] for k in KEYS}
            rows.append(dict(frame=frame, scores=scores, psnr_gap=gaps['PSNR'], masked_psnr_gap=gaps['masked_PSNR'],
                min_gap=min(gaps.values()), mean_gap=float(np.mean(list(gaps.values()))),
                strongest_other=strongest, wins_both_all=min(gaps.values()) > 0,
                own_psnr_percentile=float(ranks[i, 0]), own_masked_percentile=float(ranks[i, 1]),
                own_min_percentile=float(ranks[i].min()), own_mean_percentile=float(ranks[i].mean()),
                high_own_scores=bool(np.all(own_scores[i] >= medians)), mask_fraction=stat['area']))
        # Audit saved-image metrics against the full-scene report for all four methods.
        mean_checks = {}
        for method in fig.METHODS:
            mean_checks[method] = {}
            for key in KEYS:
                mean = float(np.mean([r['scores'][method][key] for r in rows]))
                error = mean-report['metrics'][method][name][key]
                if abs(error) > 1e-4:
                    raise ValueError(f'Full-scene score disagrees with report: {name}/{method}/{key}: {error}')
                mean_checks[method][key] = dict(recomputed=mean, reported=report['metrics'][method][name][key], delta=error)
        eligible = sorted((r for r in rows if r['high_own_scores']),
                          key=lambda r: (-r['min_gap'], -r['mean_gap'], -r['own_min_percentile'], r['frame']))
        fallback = sorted((r for r in rows if not r['high_own_scores']),
                          key=lambda r: (-r['own_min_percentile'], -r['own_mean_percentile'], -r['min_gap'], r['frame']))
        ranked = eligible+fallback
        chosen = ranked[:3]
        if len(chosen) != 3:
            raise ValueError('Fewer than three common views: '+name)
        summary = dict(scene=name, all_views=len(rows), high_own_views=len(eligible),
            all_dual_wins=sum(r['wins_both_all'] for r in rows),
            high_own_dual_wins=sum(r['wins_both_all'] for r in eligible),
            primary=chosen[0]['frame'], primary_dual_win=chosen[0]['wins_both_all'],
            selected_high_own_count=sum(r['high_own_scores'] for r in chosen),
            own_PSNR_median=float(medians[0]), own_masked_PSNR_median=float(medians[1]))
        summaries.append(summary)
        for rank, row in enumerate(ranked, 1):
            flat = dict(scene=name, selection_rank=rank, frame=row['frame'], selected=rank<=3,
                **{f'{m}/{k}': row['scores'][m][k] for m in fig.METHODS for k in KEYS},
                **{k: row[k] for k in ('psnr_gap', 'masked_psnr_gap', 'min_gap', 'mean_gap',
                   'own_psnr_percentile', 'own_masked_percentile', 'high_own_scores', 'wins_both_all', 'mask_fraction')})
            all_csv.append(flat)
            if rank <= 3:
                selected_csv.append(flat)
        for rank, row in enumerate(chosen, 1):
            fig.aligned_images(roots, row['frame'])
            values = row['scores']
            lines.append(f'| {name} | {rank} | {row["frame"]} | '+
                ' | '.join(f'{values[m]["PSNR"]:.4f} / {values[m]["masked_PSNR"]:.4f}' for m in fig.METHODS)+
                f' | {row["psnr_gap"]:+.4f} | {row["masked_psnr_gap"]:+.4f} | '+('是' if row['high_own_scores'] else '否，仅补充')+' |')
        selections[name] = dict(frames=[r['frame'] for r in chosen], source_test_frames=len(frames),
            scoring_source=str(source), scoring_source_sha256=fig.sha(source),
            per_frame_metrics={r['frame']: r['scores'] for r in chosen}, selected_ranks=chosen,
            correction_labels=labels, summary=summary,
            official_mask_sources={r['frame']: masks[Path(r['frame']).stem] for r in chosen})
        fig.save_json(out/f'{name}_scores.json', dict(scene=name, roots={m: str(p) for m,p in roots.items()},
            scoring_source=str(source), source_sha256=fig.sha(source), full_scene_checks=mean_checks,
            summary=summary, ranked_views=ranked))
        print(name, summary, flush=True)
    fig.save_json(out/'selected_frames.json', dict(created_at=datetime.now().astimezone().isoformat(),
        kind='relative_joint_psnr', protocol=PROTOCOL, checkpoint=5000,
        report_source=str(job/'comparison.json'), report_sha256=fig.sha(job/'comparison.json'),
        scored_test_views=len(all_csv), scenes=selections))
    fig.save_json(out/'summary.json', dict(scenes=summaries, primary_dual_win_scenes=sum(r['primary_dual_win'] for r in summaries)))
    for name, records in [('all_views_ranked.csv', all_csv), ('selected_view_metrics.csv', selected_csv)]:
        with (out/name).open('w', newline='') as handle:
            writer=csv.DictWriter(handle, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines += ['', '## 有无全面领先视角的检查', '',
              '| 场景 | 全部视角 | 自身两项较高 | 全部双指标领先 | 较高区间内双指标领先 | 主视角双指标领先 |',
              '|---|---:|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f'| {r["scene"]} | {r["all_views"]} | {r["high_own_views"]} | {r["all_dual_wins"]} | {r["high_own_dual_wins"]} | '+('是' if r['primary_dual_win'] else '否')+' |')
    lines += ['', f'共核对 {len(all_csv)} 个相机、{len(all_csv)*4} 张实际渲染；选择 {len(selected_csv)} 个共同视角。', '',
              '[所有视角分数及排名](all_views_ranked.csv) · [入选视角四方法指标](selected_view_metrics.csv) · [原先仅按自身双指标择优](../../qualitative_pm5000_joint_psnr/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
