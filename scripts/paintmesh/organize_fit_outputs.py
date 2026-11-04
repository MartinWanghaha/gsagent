#!/usr/bin/env python3
"""Publish completed fit/fit-fix runs as best_results-style relative-link indexes."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path


METRICS = ('PSNR', 'masked_PSNR', 'SSIM', 'masked_SSIM', 'LPIPS', 'masked_LPIPS', 'FID')
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def record(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size)


def link(source, target):
    if not source.exists():
        raise FileNotFoundError(source)
    relative = os.path.relpath(source, target.parent)
    if target.is_symlink() and os.readlink(target) == relative:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError('Refusing to replace an existing entry: ' + str(target))
    target.symlink_to(relative, target_is_directory=source.is_dir())


def table(rows, leading=('场景',)):
    header = list(leading) + list(METRICS)
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(leading) + ['---:'] * len(METRICS)) + '|']
    for label, metrics in rows:
        labels = [label] if isinstance(label, str) else list(label)
        lines.append('| ' + ' | '.join(labels + [f'{metrics[k]:.6f}' for k in METRICS]) + ' |')
    return '\n'.join(lines)


def inspect(base, experiment):
    root = base / experiment
    req = read(root / 'request.json')
    comparison = read(root / 'comparison.json')
    method = f'PaintMesh {experiment} 5000'
    items = []
    for spec in sorted(req['scenes'], key=lambda x: x['scene']):
        name = spec['scene']
        scene = root / 'scenes' / name
        run = scene / 'inpaint'
        model = run / 'inpainted_3dgs'
        evaluation = model / 'evaluation/inpaint360_r4'
        result = read(evaluation / 'results.json')
        final = read(run / 'inpaint_manifest.json')
        config = read(run / 'edgs_joint/config.resolved.json')
        assert read(scene / 'status.json')['state'] == 'complete', scene
        assert result['complete'] and final['complete'], scene
        assert result['inputs']['iteration'] == config['iterations'] == 5000, scene
        assert result['inputs']['resolution'] == 4, scene
        assert config['init']['max_points'] is None, scene
        expected = set(result['inputs']['frames'])
        assert len(expected) == result['frame_count'] == spec['frames'], scene
        for folder in ('renders', 'gt', 'masks'):
            names = {p.stem for p in (evaluation / folder).glob('*.png')}
            assert names == expected, (scene, folder, names ^ expected)
        ply = model / 'point_cloud/iteration_5000/point_cloud.ply'
        published = final['outputs']['inpainted_gaussian_ply']
        assert ply.resolve() == Path(result['inputs']['model']['path']).resolve()
        assert published['sha256'] == result['inputs']['model']['sha256']
        assert published['size_bytes'] == ply.stat().st_size
        baseline_row = next(r for r in comparison['rows']
                            if r['scene'] == name and r['method'] == method)
        for k in METRICS:
            assert math.isfinite(result['metrics'][k])
            assert abs(baseline_row[k] - result['metrics'][k]) < 1e-10, (scene, k)
        camera = Path(spec['camera']['path'])
        lama = read(run / 'manifests/lama_completion_manifest.json')
        lama_input = Path(lama['input_manifest']['path'])
        assert read(camera)['frame_count'] == len(lama['frames']) == spec['virtual_count']
        links = {
            'point_cloud.ply': ply,
            'mesh.ply': model / 'mesh/ours_5000/tsdf_fusion_post.ply',
            'mesh': run / 'inpainted_mesh',
            'model': model,
            'run': run,
            'scene_run': scene,
            'evaluation': evaluation,
            'results.json': evaluation / 'results.json',
            'renders': evaluation / 'renders',
            'gt': evaluation / 'gt',
            'masks': evaluation / 'masks',
            'train_renders': model / 'train/ours_5000/renders',
            'training_config.json': run / 'edgs_joint/config.resolved.json',
            'requested_config.json': scene / 'config.json',
            'camera.json': camera,
            'camera_source.json': Path(spec['camera_source']['path']),
            'tracker': Path(spec['removal_root']) / 'tracker',
            'trajectory.png': Path(spec['removal_root']) / 'tracker/trajectory.png',
            'lama_input': run / 'lama/input',
            'lama_output': run / 'lama/output',
            'lama_input_manifest.json': lama_input,
            'lama_manifest.json': run / 'manifests/lama_completion_manifest.json',
            'initialization': run / 'edgs_init',
            'initialization_manifest.json': run / 'manifests/edgs_init_manifest.json',
            'joint_manifest.json': run / 'manifests/edgs_joint_manifest.json',
            'local_selection_manifest.json': run / 'manifests/local_selection_manifest.json',
            'inpaint_manifest.json': run / 'inpaint_manifest.json',
            'training_view.jpg': run / 'edgs_joint/debug/final.jpg',
            'logs': scene / 'logs',
            'receipts': scene / 'receipts',
        }
        if experiment == 'fit-fix':
            links.update(lama_original=run / 'lama_original/output',
                         difix=run / 'difix',
                         difix_raw=run / 'difix/raw',
                         difix_preview=run / 'difix/preview.jpg')
        if not config['local_background']['enabled']:
            links.pop('local_selection_manifest.json')
        for target in links.values():
            if not target.exists():
                raise FileNotFoundError(target)
        items.append(dict(scene=name, spec=spec, result=result, config=config,
                          final=final, links=links, eval_record=record(evaluation / 'results.json')))
    assert len(items) == 11 and sum(i['result']['frame_count'] for i in items) == 425
    return root, req, comparison, items


DIRECTORY_GUIDE = '''| 入口 | 内容 |
|---|---|
| `point_cloud.ply` | 最终 5000 步高斯模型，可用于高斯查看器 |
| `mesh.ply` | 后处理后的 PGSR TSDF 三角网格 |
| `mesh/` | 完整语义 mesh、标签、置信度及说明 manifest |
| `model/` | 完整发布模型，包含配置、训练渲染、测试评估、mesh |
| `renders/`、`gt/`、`masks/` | 全部 test 视角的预测、GT、评估 mask，同名一一对应 |
| `evaluation/`、`results.json` | 完整评估产物；逐视角指标在 `results.json` 的 `per_view` |
| `train_renders/` | 保存的训练视角渲染，与 test 渲染分开 |
| `training_config.json`、`requested_config.json` | 训练时实际解析的配置、请求配置 |
| `camera.json`、`camera_source.json`、`trajectory.png` | 虚拟相机、来源 test 位姿及轨迹图 |
| `tracker/` | 虚拟视图跟踪结果、标注记录及 mask |
| `lama_input/` | 补全输入，包含 color/depth/normal 及对应 mask |
| `lama_output/` | 实际训练使用的补全目标，含 color/depth/normal |
| `initialization/`、`initialization_manifest.json` | 初始化点云及来源记录，区别于训练后模型 |
| `local_selection_manifest.json` | 局部旧高斯更新名单的来源；仅开启局部更新的场景具有该文件 |
| `lama_input_manifest.json`、`lama_manifest.json`、`joint_manifest.json` | 补全输入、实际训练目标与训练输出溯源 |
| `inpaint_manifest.json` | 最终高斯、mesh、语义产物的身份与哈希 |
| `training_view.jpg` | 已保存的最后训练调试图 |
| `selection.json` | 本发布目录采用的固定 5000 步结果及来源，没有重新择优 |
| `logs/`、`receipts/` | 各阶段日志与执行记录 |
| `run/`、`scene_run/` | 原始 inpaint 工作区、原始场景完整工作区 |
'''


def publish(base, experiment, inspected):
    root, req, comparison, items = inspected
    out = base / (experiment + '_release')
    if out.exists():
        marker = out / 'release.json'
        if not marker.exists() or read(marker).get('source_root') != str(root):
            raise FileExistsError('Refusing to reuse an unrelated directory: ' + str(out))
    out.mkdir(exist_ok=True)
    created = datetime.now().astimezone().isoformat()
    write(out / 'release.json', dict(source_root=str(root), experiment=experiment,
                                   created_at=created, complete=False, layout='best_results-style',
                                   storage='relative_symlinks', iteration=5000))
    # Small reports are snapshots; model/image artifacts use relative symlinks.
    for filename in ('comparison.md', 'comparison.csv', 'comparison.json',
                     'TRAINING_PLAN.zh-CN.md', 'request.json', 'status.json'):
        shutil.copyfile(root / filename, out / filename)
    for name in ('code_snapshot', 'baseline_snapshot'):
        link(root / name, out / name)
    link(REPO / 'scripts/paintmesh/organize_fit_outputs.py', out / 'organize_fit_outputs.py')
    configurations = {}
    selections = []
    scene_rows = []
    for item in items:
        name, spec, result, cfg = (item[k] for k in ('scene', 'spec', 'result', 'config'))
        target = out / name
        target.mkdir(exist_ok=True)
        for key, source in item['links'].items():
            link(source, target / key)
        selection = dict(scene=name, experiment=experiment, iteration=5000,
                         policy='Fixed completed 5000-step checkpoint; no per-scene best selection',
                         frame_count=result['frame_count'], metrics=result['metrics'],
                         evaluation_record=item['eval_record'], protocol=result['protocol'],
                         gaussian=item['final']['outputs']['inpainted_gaussian_ply'],
                         mesh=item['final']['outputs']['pgsr_mesh'],
                         links={k: os.readlink(target / k) for k in item['links']})
        write(target / 'selection.json', selection)
        selections.append(selection)
        configurations[name] = dict(config=cfg, mask_dilation=spec['mask_dilation'],
                                    target_ids=spec['target_ids'], virtual_count=spec['virtual_count'],
                                    source_record=record(item['links']['training_config.json']))
        scene_rows.append((f'[{name}]({name}/README.md)', result['metrics']))
        text = [f'# PaintMesh {experiment} / {name}', '',
                f'固定 5000 步，1/4 分辨率；完整测试集 {result["frame_count"]} 帧。', '',
                table([('5000', result['metrics'])], ('步数',)), '',
                f'对象 ID：{spec["target_ids"]}；MASK_DILATION={spec["mask_dilation"]}；虚拟帧 {spec["virtual_count"]}。',
                f'初始化 depth={cfg["init"]["use_depth"]}、normal={cfg["init"]["use_normal"]}、'
                f'surface_init={cfg["surface_init"]["mode"]}、grid_pixels={cfg["surface_init"]["grid_pixels"]}；init.max_points=null。',
                f'局部旧点更新={cfg["local_background"]["enabled"]}；背景冻结={cfg["freeze_background"]}；'
                f'剪枝={cfg["pruning"]["enabled"]}；自适应增密={cfg["adaptive_density"]["enabled"]}。', '',
                '[实际训练配置](training_config.json) · [逐视角指标](results.json) · [发布来源](selection.json) · [全场景评估](../comparison.md)', '',
                DIRECTORY_GUIDE]
        if experiment == 'fit-fix':
            text += ['`lama_original/` 是 Difix 之前的独立 LaMa 输出；`difix_raw/` 是 Difix 全图原始输出；'
                     '`lama_output/color/` 才是 mask 内合成后的实际训练 RGB。深度、法线沿用原补全结果。'
                     '处理记录见 `difix/`，预览见 [difix_preview](difix_preview)。', '']
        text += ['大文件通过相对符号链接引用原实验。移动到其他机器时需解引用链接或同时迁移其目标文件。', '']
        (target / 'README.md').write_text('\n'.join(text))
    macro = {k: sum(i['result']['metrics'][k] for i in items) / len(items) for k in METRICS}
    weighted = {k: (macro[k] if k == 'FID' else
                   sum(i['result']['metrics'][k] * i['result']['frame_count'] for i in items) / 425)
                for k in METRICS}
    write(out / 'metrics.json', dict(scene_count=11, frame_count=425,
                                    scenes={i['scene']: i['result']['metrics'] for i in items},
                                    scene_macro=macro, frame_weighted_fid_scene_mean=weighted))
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'frame_count', *METRICS])
        writer.writeheader()
        for i in items:
            writer.writerow(dict(scene=i['scene'], frame_count=i['result']['frame_count'], **i['result']['metrics']))
        writer.writerow(dict(scene='11_scene_macro', frame_count=425, **macro))
        writer.writerow(dict(scene='425_frame_weighted_FID_scene_mean', frame_count=425, **weighted))
    write(out / 'configurations.json', configurations)
    write(out / 'selection.json', selections)
    doc = [f'# PaintMesh {experiment} 发布结果', '',
           '11/11 场景、425 张测试图；全部使用本实验固定的 5000 步模型。目录形式与 `best_results` 一致，按场景直接访问。', '',
           '[逐场景及基线完整对比](comparison.md) · [指标 CSV](metrics.csv) · [完整实际配置](configurations.json) · '
           '[训练方案](TRAINING_PLAN.zh-CN.md) · [完整性检查](validation.json)', '',
           '虚拟相机由数据集 `test*` 相机位姿插值得到；训练未使用 test RGB 或 test mask。'
           '这是使用测试位姿的实验协议，与其他三个方法的训练视角协议有差异。', '',
           ('fit 使用原先独立 LaMa 补全。' if experiment == 'fit' else
            'fit-fix 在相同独立 LaMa 输出上应用 Difix；仅将 mask 内的 RGB 更新到训练目标，洞外及深度、法线保持原来源。'),
           '所有场景 init.max_points=null。redbull 的 depth_anchor 网格由 32 调至 8；其他实际配置逐场景记录在 configurations.json。', '',
           table(scene_rows + [('11 场景宏平均', macro), ('425 帧加权（FID 场景均值）', weighted)]), '',
           'PSNR/SSIM 越高越好，LPIPS/FID 越低越好。masked 指标先将 mask 外置零，再计算全图指标；'
           '不是仅按 mask 像素个数归一化。FID 按场景计算，表中的汇总为场景 FID 平均，并非合并 425 帧重新计算的 FID。', '',
           '## 每个场景的目录入口', '', DIRECTORY_GUIDE,
           '根目录 comparison.md/csv/json 保留原实验与本地基线的完整指标；metrics.csv/json 只含本发布实验。'
           'selection.json 记录来源，configurations.json 汇总实际训练配置，request.json 和 status.json 是发布时快照。', '',
           '模型、mesh、图像和日志使用相对符号链接，说明与指标汇总是独立小文件；没有复制大量原始数据。'
           '下载独立副本时需解引用链接。不能删除原始 fit/fit-fix 及其上游数据后仍期待这些链接有效。', '']
    if experiment == 'fit-fix':
        doc += ['额外入口：各场景的 `lama_original/`、`difix_raw/`、`difix/`、`difix_preview`，分别用于检查原补全、'
                '修补原图、推理记录及预览；实际训练目标仍统一在 `lama_output/`。', '']
    (out / 'README.md').write_text('\n'.join(doc))
    links = [p for p in out.rglob('*') if p.is_symlink()]
    broken = [str(p) for p in links if not p.exists()]
    absolute = [str(p) for p in links if os.path.isabs(os.readlink(p))]
    assert not broken and not absolute, (broken, absolute)
    write(out / 'validation.json', dict(checked_at=created, complete=True, scene_count=11,
          test_frames=425, image_pairs_checked=425, relative_symlinks=len(links),
          broken_links=broken, absolute_links=absolute,
          model_checks='Evaluation/publish manifest hashes agree; linked path and recorded byte size verified. No full PLY rehash.',
          metric_checks='All seven per-scene metrics match original comparison rows; macro and weighted means recalculated.',
          training_changed=False, source_outputs_modified=False))
    write(out / 'release.json', dict(source_root=str(root), experiment=experiment,
          created_at=created, complete=True, layout='best_results-style', storage='relative_symlinks',
          iteration=5000, scenes=[i['scene'] for i in items], frame_count=425,
          source_comparison=record(root / 'comparison.json')))
    return dict(directory=str(out), scenes=len(items), frames=425, links=len(links), complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=REPO / 'output/paintmesh/Inpaint360GS')
    args = parser.parse_args()
    base = args.base.resolve()
    # Validate both full experiments before creating either publication.
    inspections = {e: inspect(base, e) for e in ('fit', 'fit-fix')}
    for experiment, inspected in inspections.items():
        print(json.dumps(publish(base, experiment, inspected), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
