# PaintMesh 在 Inpaint360GS 上的推荐运行配置

> **2026-10-27 全场景复查更新：** 本文的 10000 步推荐来自历史 bag 固定输入消融。本次全场景重新生成输入后，11 个场景的 PSNR 均以已保存的 5000 步快照更高，且发现部分场景的目标误删、mask 侵入与局部旧点许可范围过大。当前配置位置、逐场景参数、证据图和建议见 [全场景配置与质量复查](INPAINT360_CURRENT_CONFIG_AUDIT.zh-CN.md)。下文保留历史依据，不能把旧准备状态或 bag 最优值当作当前全场景结论。

2026-10-25 根据历史 bag 消融选出的主配置是 **原始混合初始化＋原始外观/几何监督＋局部旧点更新/远处冻结＋仅剪枝＋10000步**。
它来自已有 bag 消融结果，而不是把不同实验的优点拼成一组未经训练的参数。适用入口是本仓库 `scripts/paintmesh` 的 EDGS-PGSR 路径。

这是我们方法的运行配置，不是把我们的实现改名为官方 Inpaint360GS。bag 是用于配置选择的开发场景；目前没有证据证明任何配置在全部11场景、七项RGB指标和mesh几何质量上同时最优。

2026-10-25 按用户要求，主配置的 `init.max_points` 改为 `null`，取消初始化总点数及其关联候选池资源上限；匹配置信度、几何筛选与空间去重仍按原参数执行。下面的历史指标及来源快照保留当时配置，当前配置与历史版本的这一差异记录于来源 JSON 的 `resource_limit_update`。

## 选择依据

直接核对了 bag 下35份完整 `evaluation/inpaint360_r4/results.json`，均为同一58张真实测试相机、1/4分辨率、官方 `unseen_mask` 和相同评估定义。完整记录保存在：

- [35组指标与配置选择](../../output/evaluation/paintmesh_bag_configuration_20261025/comparison.md)；实际仓库路径为 `output/evaluation/paintmesh_bag_configuration_20261025/`。
- 该目录的 `bag_runs.csv` 保存所有候选；`selection.json` 保存模型/结果哈希、各指标最优和当前代码与历史实现的差异。
- [配置来源快照](configs/inpaint360_recommended_provenance.json) 保存推荐配置与实际初始点云、LaMa、相机、局部旧点名单的来源。

| 已完成的 bag 实验 | PSNR ↑ | masked PSNR ↑ | SSIM ↑ | masked SSIM ↑ | LPIPS ↓ | masked LPIPS ↓ | FID ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始混合，5000步 | 27.485006 | 35.415835 | 0.868939 | 0.987688 | 0.108684 | 0.007163 | 17.414529 |
| **推荐：原始混合，10000步** | **27.496404** | **35.464743** | 0.867988 | 0.987526 | 0.108247 | 0.006593 | **16.757742** |
| 原始混合，20000步 | 27.494996 | **35.548535** | 0.866834 | 0.987320 | 0.108406 | **0.006352** | 17.213125 |
| depth-anchor，10000步 | 27.498867 | 35.443731 | **0.869119** | **0.987982** | 0.108784 | 0.007317 | 17.458594 |

10000步混合配置的FID为这35组中的最低值；相对同初始化20000步配置，masked PSNR仅少约0.084 dB，训练预算减半，SSIM、LPIPS和FID更好。它是综合RGB表现与成本的选择，不是七项指标全部获胜。

在这35组历史 bag 固定输入实验中，混合配置20000步的masked PSNR最高；这不是当前全场景的训练时长推荐。剪枝截止仍保持15000，不应顺便修改剪枝窗口并把新组合当作已测结果。这35组最高masked PSNR为35.548535。

depth-anchor 作为独立可选配置保留：它修正初始化中心与采样，不依赖将补全面固定为平面。但它在上述实测中没有全面改善RGB指标，也没有完成mesh定量验证。旧的 `surface_init.mode=plane` 已从当前代码移除，不作为新实验配置。

## 固定的训练参数

主配置：[inpaint360_balanced_10000.yaml](configs/inpaint360_balanced_10000.yaml)。
可选配置：[inpaint360_depth_anchor_10000.yaml](configs/inpaint360_depth_anchor_10000.yaml)。

两份文件均由对应已评估实验的 `config.resolved.json` 经当前loader规范化后保存，避免仅依赖可变的默认配置。

| 模块 | 主配置 |
|---|---|
| 数据 | 1/4分辨率；`EVAL=false`；所有非 `test_*` 图训练；test图只评估 |
| 基础重建/实例蒸馏 | EDGS-PGSR 30000步；语义2000步；seg禁克隆/分裂；语义数据放CPU |
| 删除 | 本次模型的目标ID；`REMOVAL_THRESHOLD=0.7`，与历史bag保持一致 |
| 虚拟视角 | PGSR后端；adaptive revision 3；180帧连续开放双圈扫掠；倾角30–50° |
| LaMa | RGB/depth/normal共用已检查洞mask；bag膨胀30；`RECURSIVE_GUIDE=false` |
| 新点初始化 | RoMa＋depth混合；depth开启、normal定向关闭；confidence=0.7；重投影2px；cycle=3px |
| 表面校正 | 主配置 `surface_init.mode=none`；可选配置为 `depth_anchor` |
| 更新范围 | 新点＋许可的局部旧点；远处旧点严格冻结；不启用全点训练/末尾写回 |
| 密度 | 只剪枝，不clone/split；opacity阈值0.005；截止15000步 |
| 外观 | `appearance.mode=masked`；洞内RGB=1、洞外背景保持=1、SSIM=0.2、masked LPIPS=0.005 |
| 几何/覆盖 | coverage=0.1、depth=0.1、normal=0.05、depth-normal=0.01、multiview=0.01 |
| 调度 | 10000步；seed=0；几何从100步开始，2000步渐增；scale-flatten=0 |
| 学习率 | position=0.00016、scale=0.005、rotation=0.001、feature=0.0025、opacity=0.05 |
| 局部旧点 | mask膨胀10、至少2视角支持、距离因子3、Gaussian半径上限因子6 |
| mesh | 运行inpaint至Stage8；PGSR深度＋TSDF后处理；voxel=0.002、max_depth=5、clusters=1 |

`freeze_background=true` 与 `local_background.enabled=true` 一起使用，含义是允许名单内的局部旧点更新，远处旧点冻结，不是全部旧点都冻结。外观监督的RGB洞外目标是删除后的背景渲染；SSIM比较渲染与completed RGB；masked LPIPS使用洞区域并保留上下文，不切换到官方式四patch外观。

这里没有通过bag实测证明开启ADC、改为官方式外观、打开normal初始化、放开全部旧点或权重100的尺度扁平化更优，因此没有把它们加入推荐主配置。历史ADC五组是在官方式外观下进行，不能据此声称所有外观配置下的ADC都更差。

## 直接用于已有 bag removal

加载环境配置不会启动训练：

```bash
cd /home/member/wyc/gsagent
export PAINTMESH_INIT_MODE=mixed
source scripts/paintmesh/configs/inpaint360_recommended.env
export GPU=1
```

复用已有的180帧 adaptive removal 和已完成跟踪，在独立名称下运行完整inpaint Stage1–8：

```bash
removal_root="$PWD/output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_adaptive180_sweep"
REMOVAL_ROOT="$removal_root" \
  INPAINT_RUN_ROOT="$removal_root/inpaint/$INPAINT_RUN_NAME" \
  END_STAGE=8 bash scripts/paintmesh/run_inpaint.sh \
  Inpaint360GS/inpaint360 bag 4 36 none 1
```

输出为 `.../table1_adaptive180_sweep/inpaint/recommended_mixed_10000/`。该命令重新执行/验证公共补全与当前初始化，因此是同参数的新流程实验，不保证逐位复现历史模型。旧消融的10000步结果复用了固定的初始化、LaMa、相机及79,196个可更新旧点名单。

若要选depth-anchor，使用独立命名的同参数实验：

```bash
export PAINTMESH_INIT_MODE=depth_anchor
source scripts/paintmesh/configs/inpaint360_recommended.env
removal_root="$PWD/output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_adaptive180_sweep"
REMOVAL_ROOT="$removal_root" \
  INPAINT_RUN_ROOT="$removal_root/inpaint/$INPAINT_RUN_NAME" \
  END_STAGE=8 bash scripts/paintmesh/run_inpaint.sh \
  Inpaint360GS/inpaint360 bag 4 36 none 1
```

输出为 `.../inpaint/recommended_depth_anchor_10000/`，不要用主配置的FID作为该配置的结果。

## 全部11场景的固定实验协议

场景为 bag、car、cone_red、cone_yellow、cube、doppelherz、fruits、garden_toys、redbull、toys、truck。
当前11场景的语义模型均已存在；新输出根或新数据才需要从头执行seg。`test_*` 的用途沿用[官方数据集说明](https://github.com/dfki-av/Inpaint360GS#dataset-structure)，不参与训练、LaMa参考或初始化。

2026-10-25 整理时的前置产物状态：bag 已有本配置对应的180帧adaptive removal；其余场景当时需要新建180帧虚拟视图和跟踪。此后全场景运行已经完成，当前产物和质量问题以[2026-10-27复查](INPAINT360_CURRENT_CONFIG_AUDIT.zh-CN.md)为准。不能只对旧90帧结果设置 `VIRTUAL_CAMERA_COUNT=180` 就把它变成同一组实验。

```bash
cd /home/member/wyc/gsagent
export PAINTMESH_INIT_MODE=mixed
source scripts/paintmesh/configs/inpaint360_recommended.env
export GPU=1

# 从头建基础模型时使用；已有模型先核对其resolution/eval配置。
bash scripts/paintmesh/run_seg Inpaint360GS all 4 1

# 示例：先以本次语义渲染确认目标编号。
scene=bag
target_ids=36
surrounding_ids=none
removal_root="$OUTPUT_ROOT/paintmesh/Inpaint360GS/inpaint360/$scene/removal/$RUN_NAME"
REMOVAL_ROOT="$removal_root" END_STAGE=5 bash scripts/paintmesh/run_remove.sh \
  Inpaint360GS/inpaint360 "$scene" 4 "$target_ids" "$surrounding_ids" 1

# 检查整条轨迹的洞mask并完成跟踪后执行。
REMOVAL_ROOT="$removal_root" \
  INPAINT_RUN_ROOT="$removal_root/inpaint/$INPAINT_RUN_NAME" END_STAGE=8 \
  bash scripts/paintmesh/run_inpaint.sh \
  Inpaint360GS/inpaint360 "$scene" 4 "$target_ids" "$surrounding_ids" 1

# 七项指标：使用真实test相机与官方unseen_mask。
CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$PAINTMESH_ENV" \
  python scripts/paintmesh/eval_inpaint360.py run \
  --model-path "$removal_root/inpaint/$INPAINT_RUN_NAME/inpainted_3dgs" \
  --source-path "$DATA_ROOT/Inpaint360GS/inpaint360/$scene" \
  --resolution 4 --iteration 10000
```

旧八场景的目标编号记录为 bag=36、car=7、cone_red=24、cone_yellow=8、cube=25、doppelherz=18、redbull=8、truck=50，仅适用于现有这些语义模型；不能直接套用到重训模型，也不能套用官方Inpaint360GS、Gaussian Grouping或Aura的编号。全部11场景目前的实际编号已记录于 [inpaint360_full_scenes.json](configs/inpaint360_full_scenes.json)，其中 doppelherz 的 ID 18 已发现合并邻近物体，需要纠正删除选择，不能仅凭编号存在即认为标注正确。

原来8场景的LaMa膨胀视觉记录：bag/car/cone_red/cone_yellow=30、cube=20、doppelherz/redbull/truck=10，来源为 [lama_dilation_by_scene.json](configs/lama_dilation_by_scene.json)。这些记录来自旧90帧轨迹，不是新180帧轨迹上的定量最优值。环境配置默认采用bag已测的30；其他场景在训练视角上检查目标残留和阴影后，用 `MASK_DILATION=...` 显式覆盖并记入该场景manifest。fruits/garden_toys/toys 没有该历史检查记录。

运行命令要用 Bash数组/逐场景显式参数调用已有入口；不要把整个数据集统一写死为bag的目标ID。完整mask检查完成前，不应把脚本自动跑过等同于正确去物体。

## 可重复性与结果边界

1. 配置来源、历史结果模型、初始化、LaMa、相机、局部旧点名单都有SHA256记录。当前源码相对历史最佳实验有7个实现/默认配置文件变化，明细见来源快照；配置相同不等于代码和中间产物逐位相同。
2. 需要核对历史数值时，直接使用原已评估模型或固定其全部输入及对应源码；需要验证当前完整pipeline时，用上述独立输出名称重新训练，不把新初始化的结果预先填成历史最佳值。
3. 删除模型中已经存在的残留旧高斯，不会因为换一种新点初始化就自动消失；它也可能在“远处冻结”规则下保持不动。先检查删除结果和洞mask，再解释浮点是否由新增点产生。
4. 七项指标只评价3DGS的RGB渲染。以上消融没有额外运行TSDF，因此不能据此断言mesh最优；Stage7–8需要实际完成并检查。
5. 其他10场景先固定同一训练配方再评估。bag的测试视角已用于调参，报告时应把这一点与未经调参的场景区分。对照方法的训练视角数、分辨率和训练预算差异也应保留在实验说明中。

本次只整理配置、指标来源和运行命令，不自动启动新的训练或替换已有实验结果。
