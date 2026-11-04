# RGB 锚点引导的深度初始化

`edgs-pgsr` 的 `surface_init.mode: depth_anchor` 替代原 `plane` 模式。它只修改新点的中心与采样布局，不拟合平面，不改变 seg、训练损失、旧点更新权限或密度控制。默认 `none` 仍是原混合初始化；历史平面结果可读取校验，但不能再创建新的 `plane` 初始化。

## 如何初始化

1. **独立提取 RGB 几何锚点。** 保存 RoMa 双向匹配的像素对应，在任何 depth 修正之前建立三视角闭环。新匹配流程会为每个参考帧邻居中基线最宽的一对补齐第三条匹配边（原邻接图不一定包含三角形）。三个边的对应需在图像坐标中一致，再进行只使用像素重投影的三角化和鲁棒精化。检查正深度、图像范围、重投影误差、视差角与条件协方差。禁止直接把旧的 depth-refined XYZ 当成独立 RGB 锚点，也不使用三维最近邻给深度点“吸附”目标。
2. **校正偏差，不拟合背景形状。** 对每个相机，以锚点对应像素建立 `锚点 camera-z − LaMa depth` 残差。拟合双线性网格上的连续校正场，采用协方差权重和 Huber IRLS；正则只平滑校正量。原深度的曲面形状可以保留，中心沿原相机射线移动。相机相似变换的尺度参与深度单位换算。
3. **限制有效域。** 只允许 mask 内锚点约束真正可修改的像素；mask 外深度逐值保留。校正网格在无效深度、mask 边界和深度跳变处断开。每个候选像素需要同一连续分量内至少三个附近的有效锚点；无支持的区域不补满。边界零校正约束和校正量正则是明确的平滑先验，不是对真实背景的证明。
4. **多视角筛选再采样。** 在校正后的深度图上取点，至少两个有足够视差的其他相机也必须有锚点支持，并达到深度一致性比例；拒绝侵入已知前景的点。体素内选一致性更好的样本，不跨层取三维平均；剔除与保留背景重复的点。RGB 原始离群点不直接拼回结果。

颜色仍从来源 LaMa RGB 双线性采样并转 SH DC；尺度使用原混合候选的 `spacing`，单位 quaternion、opacity 的原规则不变。要求 `init.use_depth=true`、`init.use_normal=false`。原旧背景 PLY 的所有字段保持不变。

这不是对伪图像绝对精度的保证：闭环匹配可能存在共同错误，多个补全深度也可能一致地错误。输出 `confidence` 是跨视角深度通过比例，**不是几何准确率概率，也不是独立留出验证**。初始化更好也不代表最终训练不再漂移：本次不替换训练使用的原 LaMa depth/normal。

## 配置

```yaml
init:
  use_depth: true
  use_normal: false
surface_init:
  mode: depth_anchor
  track_join_pixels: 1.0
  anchor_min_angle_deg: 5.0
  anchor_max_sigma_spacing: 5.0
  min_anchor_observations: 3
  grid_pixels: 32
  anchor_radius_pixels: 64
  depth_edge_relative: 0.05
  smooth_weight: 0.3
  ridge_weight: 0.001
  candidate_pixel_step: 3
  min_cross_depth_views: 2
  cross_depth_spacing: 2.0
  min_cross_depth_fraction: 0.6
  sampling_distance_factor: 0.75
```

锚点位置方差按约 1 像素观测噪声计算，再以旧背景间距归一化；这是**给定匹配与相机正确时**的条件不确定度。允许作为校正观测的锚点并不等于最终高置信度表面点。其余 RoMa confidence、cycle、reprojection 门槛仍取 `init` 配置。

普通 `run_inpaint.sh` 可通过环境变量启用：

```bash
export EDGS_SURFACE_INIT_MODE=depth_anchor
export EDGS_INIT_USE_DEPTH=true
export EDGS_INIT_USE_NORMAL=false
```

随后执行原 `run_inpaint.sh` 命令，并指定新的 `INPAINT_RUN_ROOT`。

或者在独立 `EDGS_INPAINT_CONFIG` 中指定上面的配置。初始化与训练读取同一解析配置并校验一致性。不要直接复用已完成的旧平面/混合初始化输出目录，否则输入身份校验会阻止覆盖混用。

## 诊断与复用

`edgs_init/surface_model.json` 记录闭环数量、有效 RGB 锚点数、原始 mask 内锚点数、各帧支持范围、筛选后点数和相机模型。`surface_evidence.npz` 保存锚点、三视角像素、协方差、每帧校正网格和支持域。没有足够支持、或仅覆盖膨胀边界而未覆盖原始洞内时，也写入这两个诊断文件，状态为 `insufficient_support`，随后失败退出；不生成完成 manifest，不回退到平面或未校正深度。

成功结果的 `support.npz` 保存 `raw_depth`、`corrected_depth`、`depth_votes`、`depth_checks`、`frame/pixel_uv` 等。新点用 `source_kind=2`、`other_frame=-1` 标识为表面采样点。`parent_index` 只记录最近混合候选的溯源关系，不参与位置校正。manifest 额外校验所有输出身份、射线反投影、跨视角证据及旧背景不变。

`initialize_paintmesh_edgs.py --proposal-initialization <旧混合初始化 manifest>` 可复用身份匹配的历史候选，避免重新匹配；其余 init/matcher/seed、LaMa、相机、背景必须一致。历史文件只存筛选后的两视角对应，已经丢失的闭环无法恢复，可能比新一轮保存完整原始对应得到更少锚点；`track_source` 会明确记录这一差别。不能把先前 pilot 的少量边界点描述成洞内高置信度补全。

## 初始化单变量消融

在 GPU 1 后台顺序完成新的初始化、训练、发布和七项指标评估：

```bash
cd /home/member/wyc/gsagent
/home/member/.conda/envs/paintmesh/bin/python \
  scripts/paintmesh/run_surface_initialization_ablation.py --gpu 1 --detach
```

默认基线为 bag 的 `table1_adaptive180_sweep/inpaint/edgs_depth_local_prune_perceptual`。默认重新运行 RoMa 并补齐三视角匹配闭环，不复用旧缓存中已经丢失的对应。若已有完成且通过校验的新初始化，可显式传入 `--surface-initialization /绝对路径/manifests/edgs_init_manifest.json` 跳过匹配。

配置仅改变 `surface_init.mode: none → depth_anchor`，保持原始外观＋几何监督、迭代数、剪枝和远处冻结。初始化成功后，逐位复用基线旧点解冻名单，不按新点分布重新挑选旧点；然后才训练、发布和评估。初始化支持不足时保存诊断并停止，不降低阈值或改用其他初始化。

每次创建独立实验目录，后台进程与 SSH 终端分离。日志入口为 `logs/original_supervision_depth_anchor_init_gpu1.latest.json`；其中包括 PID、总日志、状态和输出目录。各阶段写同一个 `depth_anchor.log`。原始请求保存在 `request.json`，实际生成的新初始化、旧点名单校验证据保存在 `resolved_initialization.json`，新结果追加到对照报告，历史结果保留。恢复已有任务使用 `--resume <日志任务目录> --detach`；不要重复启动一个新实验。

## 固定 depth_anchor 初始化的训练步数消融

```bash
cd /home/member/wyc/gsagent
bash scripts/paintmesh/run_depth_anchor_iteration_ablation.sh
```

此脚本默认 GPU 1 后台运行，基线固定为已完成的 `original_supervision_depth_anchor_init_20261022_215217/depth_anchor`。按 2000、4000、7000、9000、10000、15000、20000 步依次独立训练，每组随后发布和评估七项指标。全部复用该基线的初始化 manifest/PLY、旧点更新名单、LaMa 和相机；不重新初始化，不从 5000 步成品或上一组 checkpoint 接着训练。

仅 `iterations` 变化：原始 masked 外观＋全部几何监督、匹配门槛、种子和学习率不变；几何损失仍从第 100 步开始、渐增 2000 步，剪枝仍在 15000 步截止。这些调度不会随总步数拉伸。每组精确配置差异记录在日志任务目录 `configuration_diff.json`。

新输出位于 `inpaint/original_supervision_depth_anchor_iterations_<时间>/iter_<步数>`。日志入口为 `logs/original_supervision_iteration_ablation_gpu1.latest.json`，任务目录保存 `run.log`、每组 `iter_<步数>.log` 和 `status.json`。汇总追加到原 `comparison_paper_bag/comparison.md`，同时写入新输出目录的 `comparison.md`、`summary.json` 和完成组的 `summary.csv`。表中的 5000 步对照来自同一 depth_anchor 初始化实验，论文列单独读取，避免把旧混合初始化结果当成本轮基线。

## 其他场景：同配方 5000 步批处理

```bash
cd /home/member/wyc/gsagent
/home/member/.conda/envs/paintmesh/bin/python \
  scripts/paintmesh/run_depth_anchor_scene_batch.py --gpu 1 --detach
```

使用上述 bag 的完整 5000 步配置，按 car、cone_red、cone_yellow、cube、doppelherz、redbull、truck 串行运行。每场景复用自己的 `table1_hemi90/inpaint/edgs_rgbdn_joint_woDP` 的 removal、LaMa 和 90 帧相机，重新做 depth_anchor 初始化、局部旧高斯选择、训练、发布和七项指标评估。保留原本 masked 外观＋几何监督、局部旧点更新＋远处冻结和只剪枝策略。bag 既有 5000 步结果只作为参考，不重新训练。fruits、garden_toys、toys 缺少 removal/LaMa 前置输出时会明确列为未覆盖。

该实验是把 bag 的配方应用于其他场景，不是相对这些场景早期配置的单变量消融。初始化支持不足的场景记为失败、保存诊断并继续下一场景，不放松阈值、不复用旧模型冒充新结果。新训练路径为各场景 `removal/table1_hemi90/inpaint/original_supervision_depth_anchor_5000_<时间>`。

日志入口：`logs/original_supervision_depth_anchor_5000_scenes_gpu1.latest.json`。总日志、各场景日志和状态均在该入口指向的日志目录。对照报告自动更新至 `output/evaluation/original_supervision_depth_anchor_5000_scenes_<时间>/comparison.md`，同时保存 JSON/CSV。使用 `--resume <日志目录> --detach` 恢复相同任务，完成且通过核验的场景不会重训。

论文逐场景值保存在 `configs/inpaint360_paper_table7.json`，来源为 arXiv v1 Table 7。报告列出每场景七项指标和差值，并仅对双方相同的已完成场景子集计算场景平均；另列本方法按官方代码的聚合（前六项按帧加权、FID 按场景平均）。不将七场景或含既有 bag 的八场景均值当作论文完整十一场景 Table 1 的复现。
