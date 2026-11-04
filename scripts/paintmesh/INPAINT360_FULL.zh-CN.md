# 全场景 adaptive180、5000/10000 步对照

入口：`scripts/paintmesh/run_inpaint360_full.py`。场景策略存于
`configs/inpaint360_full_scenes.json`，训练参数使用
`configs/inpaint360_balanced_10000.yaml`。

```bash
cd /home/member/wyc/gsagent
/home/member/.conda/envs/paintmesh/bin/python \
  scripts/paintmesh/run_inpaint360_full.py --detach
```

该命令创建新实验并脱离终端，固定使用物理 GPU 1，串行处理全部 11 个场景。
已有完整的 1/4 分辨率语义重建被复用；训练排除所有 `test_*` 图像，`eval=false`。
所有场景创建独立的 `removal/full_r4_adaptive180_<时间>` 和新的 180 帧跟踪，
不覆盖历史消融实验。

## 配置

| 项目 | 设置 |
|---|---|
| 虚拟视图 | PGSR、adaptive、180 帧；`configs/inpaint360_adaptive180.json` |
| 初始化 | depth=true、normal=false、surface_init=none |
| 初始化总点数 | `init.max_points: null`，不设总数及关联候选池上限；保留匹配质量筛选与空间去重 |
| 背景 | local_background.enabled=true、freeze_background=true |
| 密度 | pruning.enabled=true、adaptive_density.enabled=false |
| 监督 | 原始 masked 外观监督与深度、法线、深度法线一致性、多视图几何监督 |
| 训练 | 同一次 10000 步；5000 步中途导出、10000 步最终导出 |
| LaMa | RECURSIVE_GUIDE=false |
| 测试 | 原始 `test_*` 相机、1/4、官方 unseen_mask、七项指标 |

| 场景 | 本项目 ID | MASK_DILATION |
|---|---|---:|
| bag | 36 | 30 |
| car | 7 | 30 |
| cone_red | 24 | 30 |
| cone_yellow | 8 | 30 |
| cube | 25 | 20 |
| doppelherz | 18 | 10 |
| fruits | 44 | 10 |
| garden_toys | 14 | 20 |
| redbull | 8 | 10 |
| toys | 22,23,24 | 20 |
| truck | 50 | 10 |

ID 从本项目已有语义预测的训练图中核对，不能直接套用其他项目的 ID。
fruits 为草莓托盘；garden_toys 为树桩顶部黄色玩偶；redbull 为金属公牛；
toys 同时移除三个玩偶。训练视图核对图保存在 `logs/paintmesh_scene_selection/`。
用户未指定的四个膨胀值根据这些训练视图与接触阴影范围选择，未使用测试 GT 调参。

## 逐场景 adaptive 相机配置

原始 30–50° 双圈扫描在五个场景触发了背景支持、遮挡或碰撞检查。
2026-10-25 使用训练相机、移除后的 Gaussians 和 mesh 重新检查轨迹，
仍采用 revision 3 的连续双圈扫描，每条路径完整保留 180 帧、179 条相邻边。
目标规划 ROI 保持 8%；逐场景覆盖参数保存在
`configs/inpaint360_full_scenes.json` 的 `adaptive_options`。
新建任务会将其与基础配置合并，固化到 `<job>/camera_configs/<scene>.json`。

| 场景 | 相对规划法线的倾角 | 背景支持代理阈值 |
|---|---|---:|
| cone_red | 10–30° | 40% |
| doppelherz | 10–30° | 40% |
| fruits | 5–25° | 40% |
| garden_toys | 10–30° | 30% |
| redbull | 5–25° | 40% |

其余六个场景保留原 30–50° / 40% 配置及已完成产物。
garden_toys 的背景为树桩和植被，背景高斯的可见点比例代理为 31.7–37.7%，
原 40% 门槛拒绝所有测试过的取景范围。该场景单独采用 30% 门槛；修复 ROI 在
180 帧中均 100% 可见。其余可见性、mesh 碰撞、相机连续性与立体匹配约束均按原值执行。
这些相机参数仅使用训练几何选择，未读取测试 RGB、unseen_mask 或 LaMa 补全结果。

本次五场景认证记录：
`logs/paintmesh_full_r4_gpu1_20261025_105925/camera_retry_probe/certified/all_five_certification.json`。
原 40% 门槛下 garden_toys 的失败搜索记录与其 30% 配置下的独立认证均保留，
不会把阈值调整前的失败报告改写为成功。

## 自动标注与跟踪

`track_holes_headless.py` 从同相机的 PGSR 原场景与移除场景 RGB 差异生成移除物体的
投影范围，作为洞 mask 标注。采用仓库已有 DeAOT 跟踪器传播；逐帧检查对移除范围的覆盖
及周边漂移，发现漂移时用该帧投影范围修正标注并重置跟踪参考。
这一步不读取测试图、unseen_mask、其他方法的 mask，也不使用 LaMa 输出反过来选择 mask。

每个新 removal 的 `tracker/automatic_annotation.json` 记录逐帧覆盖率、漂移与参考帧修正；
`tracker/review/` 保存全部 180 帧的覆盖预览。文件会如实标记是否人工检查全部帧。
所有 masks 完整生成后，仍通过原 `run_remove.sh` Stage 5 校验并提交，产物不绕过会话契约。

## 快照、评估和 mesh

`EDGS_JOINT_SAVE_ITERATIONS=5000` 只增加中途 PLY 和保留行索引的序列化。
10k 训练预算、优化器、随机状态、剪枝计划保持一致；检查点不是重新独立训练的 5k 实验。
CPU 验证覆盖中途导出对后续优化/随机状态的影响和远处冻结背景保持。

为了避免训练和评估同时占用显存，5000 步快照在 10000 步训练退出后评估。
两个评估目录：

- `<inpaint_root>/inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json`
- `<inpaint_root>/inpainted_3dgs/evaluation/inpaint360_r4/results.json`

5000 步模型目录内的 `snapshot_manifest.json` 明确声明来自同一次 10000 步训练。
最终 10000 步模型通过原始发布契约，并继续执行 Stage 7/8 的 TSDF mesh 与语义后处理。
中途 5000 步只做 3DGS 指标评估，不额外提取 mesh。

## 日志、结果与断点恢复

当前任务位置可从 `logs/paintmesh_full_r4_gpu1.latest.json` 读取。任务目录包含：

- `request.json`、`resolved_config.json`：场景路径与实际配置。
- `implementation.json`：程序和配置文件的 SHA256。
- `run.pid`、`run.log`、`status.json`：后台父进程、总日志、阶段状态。
- `<scene>_<phase>.log`：每场景每阶段输出。
- `comparison.md`、`comparison.json`：自动更新的各场景与相同场景集合均值。

恢复同一任务：

```bash
/home/member/.conda/envs/paintmesh/bin/python \
  scripts/paintmesh/run_inpaint360_full.py --job logs/paintmesh_full_r4_gpu1_<时间> --detach
```

同一入口有 GPU 1 任务锁；启动失败的重复进程不会启动第二份训练。
程序/配置改变后需要先检查差异，避免把不同实现的结果混入已记录实验。
训练从 `edgs_joint/checkpoints/latest.pth` 恢复；已完成场景被跳过，单场景失败会记录并继续其他场景。

对照读取本服务器 Inpaint360GS、Gaussian Grouping、AuraFusion360 的真实评估输出。
这些值与论文公布值分别看待。Gaussian Grouping 尚未成功的场景保持缺失，不会补零。
跨方法均值只采用各列共同完成的场景；同时保留场景宏平均及官方代码式平均
（前六项按测试帧数加权，FID 按场景平均）。基线后来完成新场景时可以刷新：

```bash
/home/member/.conda/envs/paintmesh/bin/python \
  scripts/paintmesh/run_inpaint360_full.py --job logs/paintmesh_full_r4_gpu1_<时间> --report-only
```
