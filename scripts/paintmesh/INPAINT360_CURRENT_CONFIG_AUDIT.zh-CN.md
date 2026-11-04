# PaintMesh 全场景配置与质量复查（2026-10-27）

后续已执行独立修复重训，见 [原因与实验记录](../../logs/paintmesh_quality_retrain_gpu1_20261027_170222/DIAGNOSIS.zh-CN.md)、[实时结果](../../logs/paintmesh_quality_retrain_gpu1_20261027_170222/comparison.md)。下文保留重训前的基线审查，不用新实验覆盖历史配置。

本次核对的是 `paintmesh_full_r4_gpu1_20261025_105925` 的全部 11 个场景，以及报告登记的 cube mask 修正版。读取已保存的模型配置、初始化统计、局部旧点名单、LaMa 输入/输出、训练诊断和真实测试相机指标；没有修改训练输入、模型或重新启动训练。

**当前以 PSNR 为主的全场景对比，应使用已经保存的 5000 步快照。11 个场景的全图 PSNR 都高于各自的 10000 步结果。** 历史 bag 的 10000 步推荐来自另一组固定输入，不能直接推广为这次全场景的最优配置。这里给出已测基线和需要修正的环节，不声称已有跨场景、跨指标的全局最优参数。

## 配置具体在哪

| 内容 | 文件 | 用途 |
|---|---|---|
| 统一训练参数 | [inpaint360_balanced_10000.yaml](configs/inpaint360_balanced_10000.yaml) | 初始化、损失、学习率、局部更新、剪枝 |
| 各场景配置 | [inpaint360_full_scenes.json](configs/inpaint360_full_scenes.json) | 本方法物体 ID、MASK_DILATION、相机例外参数 |
| 全流程环境参数 | [inpaint360_recommended.env](configs/inpaint360_recommended.env) | 分辨率、LaMa、各阶段开关 |
| 默认虚拟相机参数 | [inpaint360_adaptive180.json](configs/inpaint360_adaptive180.json) | adaptive 180 帧；场景 JSON 可以覆盖 |
| 批处理入口 | [run_inpaint360_full.py](run_inpaint360_full.py) | 全场景流程、保存、评估与汇总 |
| 本次任务快照 | [request.json](../../logs/paintmesh_full_r4_gpu1_20261025_105925/request.json)、[resolved_config.json](../../logs/paintmesh_full_r4_gpu1_20261025_105925/resolved_config.json) | 启动参数与任务级配置 |
| 每场景实际配置与来源 | [inventory.json](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/inventory.json) | 配置全文、原路径及 SHA256、计数和逐场景诊断 |
| 最新完整指标 | [comparison.md](../../logs/paintmesh_full_r4_gpu1_20261025_105925/comparison.md) | 四方法、5000/10000 步与修复记录 |

**复现某个已完成结果时，以该场景的 `edgs_joint/config.resolved.json` 和对应输入 manifest 为准。** 当前 YAML 与任务快照不能覆盖历史运行的真实参数。各场景准确输出路径见 inventory 的 `root` / `config_source`；cube 当前采用独立修复目录，不能按旧批次路径读取它的未修复结果。

## 当前已测基线

| 模块 | 配置 |
|---|---|
| 数据 | 1/4 分辨率；非 `test_*` 图训练；真实 test 相机与官方 unseen_mask 仅评估 |
| 基础重建/语义 | PGSR 30000 步、语义蒸馏 2000 步 |
| 虚拟视角 | PGSR 渲染、adaptive 180 帧；倾角按场景覆盖 |
| LaMa | big-lama；RGB/depth/normal；`RECURSIVE_GUIDE=false` |
| 初始化 | RoMa + depth；`use_depth=true`、`use_normal=false`、`surface_init.mode=none` |
| 匹配 | confidence=0.7、cycle=3px、重投影=2px、最小夹角=0.5°；邻居 4、每对采样 8192 |
| 初始化点数 | 当前预设 `init.max_points=null`；已完成 bag 的实际快照仍是 500000，其他 10 场景为 null |
| 旧点 | `freeze_background=true`、`local_background.enabled=true`；允许名单内旧点更新，其余旧点冻结 |
| 局部旧点选择 | LaMa mask 再膨胀 10；至少 2 视角支持；距离因子 3、半径上限因子 6 |
| 密度控制 | 仅剪枝；opacity 阈值 0.005；不 clone/split；剪枝截止 15000 |
| 外观 | 原始 `appearance.mode=masked`；洞内 L1=1、洞外保持=1、SSIM=0.2、LPIPS=0.005 |
| 几何与覆盖 | depth=0.1、LaMa normal=0.05、depth-normal=0.01、multiview=0.01、alpha/coverage=0.1；scale-flatten=0 |
| 学习率 | position=0.00016、scale=0.005、rotation=0.001、feature=0.0025、opacity=0.05 |
| 调度 | seed=0；几何从 100 步开始、2000 步渐增；实际训练 10000 步并保留 5000 步快照 |
| 当前对比模型选择 | 优先已有 `inpainted_3dgs_5000`；保留 10000 步结果作为对照 |

`normal=false` 仅指初始化法线定向关闭，**训练法线监督仍开启**。`adaptive_density.max_new_points=500000` 虽留在文件里，但 ADC 被关闭，该字段不限制本次训练；它不同于实际生效的 `init.max_points`。bag 的历史上限应在实验记录中保留，不能直接标成“11 场景全部无上限”。即使 bag 最终新点数小于 500000，也不足以证明相关候选池完全未受上限影响。

这次审查没有更改运行 YAML。若下一轮统一只训 5000 步，应作为新的预算实验记录；已有 5000 步快照的数值不能预先填给一次尚未运行的重训。

## 逐场景参数与已测结果

| 场景 | 本方法目标 ID | MASK_DILATION | 相机倾角 | PSNR 5000 | PSNR 10000 | 可更新旧点占比 |
|---|---|---:|---|---:|---:|---:|
| bag | 36 | 30 | 30–50° | 27.1590 | 27.0629 | 29.4% |
| car | 7 | 30 | 30–50° | 22.2349 | 22.1052 | 42.0% |
| cone_red | 24 | 30 | 10–30° | 21.1524 | 20.9374 | 20.1% |
| cone_yellow | 8 | 30 | 30–50° | 24.4100 | 23.5029 | 68.7% |
| cube | 25 | 20 | 30–50° | 22.7506 | 22.7429 | 10.5% |
| doppelherz | 18 | 10 | 10–30° | 25.3516 | 24.7193 | 11.4% |
| fruits | 44 | 10 | 5–25° | 25.3739 | 25.2496 | 1.8% |
| garden_toys | 14 | 20 | 10–30° | 22.9793 | 22.8611 | 4.2% |
| redbull | 8 | 10 | 5–25° | 23.4445 | 23.3357 | 3.7% |
| toys | 22 / 23 / 24 | 20 | 30–50° | 26.9246 | 26.8014 | 33.3% |
| truck | 50 | 10 | 30–50° | 25.1315 | 25.0308 | 52.2% |

garden_toys 的背景支持代理门槛为 0.3，其他场景为 0.4。这些 ID 属于当前 PaintMesh 语义模型，不能照搬其他方法的 ID。MASK_DILATION 是当前运行值，并非已经证明的逐场景最优值；特别是 car 的邻近目标侵入问题仍在。

按报告的官方代码式聚合，前六项按真实测试帧数加权、FID 按场景平均：

| 指标 | 5000 | 10000 |
|---|---:|---:|
| PSNR ↑ | 24.068986 | 23.882622 |
| masked PSNR ↑ | 35.062376 | 34.767843 |
| SSIM ↑ | 0.825097 | 0.818393 |
| masked SSIM ↑ | 0.987771 | 0.987481 |
| LPIPS ↓ | 0.134024 | 0.139325 |
| masked LPIPS ↓ | 0.010006 | 0.009709 |
| FID ↓ | 44.329910 | 49.115727 |

10000 步仅在这张聚合表的 masked LPIPS 上更好。5000 步的优先级来自当前测试结果，报告时应明确这是事后模型选择，不是独立验证集选出的最优训练时长；更短步数在这组输入上尚未系统验证。

## 效果不好的原因：已确认的事实与待验证的解释

每场景已生成“原始虚拟图 → 删除后 → LaMa mask → LaMa RGB”及“同一个训练视角的 LaMa → 5000 → 10000”两张图，共 22 张，见[图像与统计索引](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/README.md)。流程图抽查 0/60/120 帧，不表示全部 180 帧均已人工检查。原自动标注记录也不能当作全帧人工验收证明。

### 1. doppelherz：非目标物体被一起删除

[流程证据图](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/doppelherz_pipeline.jpg)中，目标药盒旁的白紫色包装盒在删除阶段也消失，并留下紫色拉伸残留；随后 LaMa 将两者所在区域一起填成桌布。[原目标语义检查图](../../logs/paintmesh_scene_selection/existing_targets.jpg)中 ID 18 覆盖两件物体。[真实测试视角对照](../../logs/paintmesh_full_r4_gpu1_20261025_105925/qualitative_pm5000_showcase/scenes/doppelherz.png)则显示该白紫包装盒应当保留。

这是上游目标归属错误的直接证据。先从训练视角纠正目标实例/三维删除选择，并保护邻近包装盒，再重做相关 removal、跟踪和 LaMa。仅改变后续损失或训练时长，不能让网络在“把它删掉”的监督下正确保留它。

### 2. car：部分 mask 侵入交通锥

[流程证据图](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/car_pipeline.jpg)的第 120 帧中，删除后的两个交通锥仍存在，但膨胀后的 mask 已覆盖其底部；LaMa 将底部填掉，只留下部分锥顶。这个问题在 RGB 补全阶段已经出现。

优先在训练视角标记需保留的交通锥，修正侵入的 mask，并覆盖真实车辆残留及阴影。简单把所有视角的膨胀量统一减小，可能又漏掉阴影，不应预先称为最优修复。当前局部旧点许可比例为 42.0%，还需复查错误 mask 对更新名单的影响。

### 3. cone_yellow：补全范围较大，局部旧点实际放开了 68.7%

[流程图](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/cone_yellow_pipeline.jpg)显示，移除锥体后，部分台阶和砖面已经可见；LaMa mask 仍覆盖原物体大范围投影。平均 mask 面积从膨胀前 16.1% 增到 25.4%。旧点更新名单包含 **652998 / 950589** 个旧高斯，因此“局部更新”在此场景并不等于只动很小一块背景。

5000→10000 步，全图 PSNR 降 0.9071 dB；按真实测试 mask 分离的洞外区域 PSNR 从 24.5547 降到 23.6075。背景退化是可量化事实；过宽 mask/更新范围与它相符，但各自因果贡献尚未通过单变量实验分离。

优先检查洞内真正缺失区域、残留和阴影，并减少对可靠背景的重新生成；随后收紧旧点许可范围。不能直接把 cube 的 alpha 门槛原样套到此场景，透明残留和阴影仍需要检查。

### 4. garden_toys：训练本身没能保留 LaMa 的纹理

[同训练视角图](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/garden_toys_training.jpg)中，LaMa 目标还有细碎叶片纹理，而 5000/10000 步渲染都成为明显的绿色模糊块。因此不能把全部问题归结为测试视角差异。

初始化共 31843 个新点，其中深度来源为 28381 个；只剪枝不能补充后续缺少的支撑点。但这些数量不足以证明“点数少就是原因”。应先检查新点中心与补全深度、跨视角对应关系和有效覆盖，再做固定初始化的表示能力/损失消融。多视角 RGB/depth 伪目标不一致也是候选原因，当前尚未完成几何一致性定量验证。

### 5. fruits / redbull：训练视角较接近 LaMa，但测试洞内仍有误差

[fruits 训练视角](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/fruits_training.jpg)、[redbull 训练视角](../../logs/paintmesh_full_r4_gpu1_20261025_105925/configuration_audit_20261027/redbull_training.jpg)表明，固定虚拟视角的外观拟合明显好于 garden_toys。真实测试修复区域却仍有色块/纹理误差；增加到 10000 步也没有提高这两者的 masked PSNR。

这更需要排查多视角补全目标、深度初始化与真实测试视角之间的几何一致性。现有 `surface_init=none` 不对中心做表面校正；渲染深度、法线损失也不等于逐个高斯中心必须位于一个正确表面。**这是解释候选，不是已确认某个 PLY 存在浮点的结论。** 本次没有做逐点几何归因。

### 6. cube：当前指标已经包含修复，旧结果不能混用

原 mask 误覆盖有效桌面时，PSNR 5000/10000 为 18.0578/17.3456；修复后为 **22.7506/22.7429**。可更新旧点占比也从约 80.4% 降到 10.5%。这次汇总与诊断使用修正版。

这提供了实际证据：上游 mask 改正能显著改善结果。它不证明其他场景都该使用同一种 mask 阈值。

### 7. cone_red、bag、toys、truck

cone_red 的 LaMa mask 平均占 28.3%，跨越蹦床边缘等结构；抽查图可见补全边界/纹理误差。多种方法在本场景的全图 PSNR 都偏低，需再用第一阶段模型在相同真实测试相机上的背景渲染分解误差，不能只归因于 inpaint。该额外渲染尚未执行。

bag 历史 10000 步 PSNR=27.4964，当前为 27.0629。历史消融固定的是旧相机、LaMa、初始化及 **79196 个可更新旧点**；当前新流程重新生成输入，可更新旧点为 **319135 个**。超参数相似并不构成同输入复现，不能把差值归因于单个损失权重。

toys/truck 的主流程抽查未发现上述程度的目标误删，但这不代表全帧无问题。truck 也有 52.2% 旧点允许更新，需注意背景保护范围。其 10000 步 masked PSNR 略有改善，而全图 PSNR 下降，说明指标目标存在取舍。

## 为什么“远处冻结”仍会背景退化

当前代码在 [select_paintmesh_local_background.py](../../submodules/EDGS/tools/select_paintmesh_local_background.py) 中，把新点中心和被删物体中心作为三维支持；在 [paintmesh_joint_region.py](../../submodules/EDGS/source/paintmesh_joint_region.py) 中，对已有膨胀的 LaMa mask 再膨胀 10 像素，检查高斯投影包围范围是否相交。满足三维距离条件且在至少 2/180 视角中相交，即可进入更新名单；这里没有逐像素遮挡可见性验证。

因此，名单之外的旧点冻结机制可以正常工作，同时名单仍包含过多背景点。尤其是较大的高斯、较宽 mask 或偏离表面的新点支持，可能扩大许可范围。应先检查名单空间分布与遮挡关系，再调整 `min_views`、距离及半径参数；不能随意指定一组新数字并称为已验证最优。

外观监督中，洞外 RGB 保持只约束所采样虚拟视角；SSIM 又比较完整 LaMa RGB。它们不能保证所有真实测试视角的背景保持不变。几何监督也继承补全 depth/normal 的不确定性。

## 下一轮最有价值的顺序

1. 固定当前 5000 步结果作为对照，保留 10000 步结果和完整配置。
2. 先修 doppelherz 的目标归属、car 的邻近交通锥保护，并逐帧检查关键场景的跟踪 mask。
3. 对 cone_yellow 检查有效洞范围及局部旧点名单，修正后保持当前原始外观+几何监督做单变量实验。
4. 对 garden_toys、fruits、redbull 检查初始化/深度的跨视角一致性；在确认输入后，再分别实验表示能力、监督和密度策略。
5. 用同一组真实 test 相机报告完整指标，区分洞内/洞外误差。查看高 PSNR 展示图不能替代全测试集检查。所有后续训练标注应来自非 test 图，GT 仅用于诊断与评估。

本次所有场景都有 10000 步产物，保存的 100 个诊断点/场景的总 loss 均为有限值；没有证据把这些质量问题解释为此次训练中断或 NaN。更可能的瓶颈集中在已发现的上游误删/错 mask，以及尚需验证的几何和多视角拟合问题。
