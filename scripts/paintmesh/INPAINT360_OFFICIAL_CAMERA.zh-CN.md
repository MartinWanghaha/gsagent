# 对齐 Inpaint360GS 相机与 mask 膨胀的全场景重跑

入口：`run_inpaint360_official_camera.py`。2026-10-30 批次为
`logs/paintmesh_circle30_d10_gpu1_20261030_114053`。

所有 11 个场景采用官方 `generate_ellipse_path(..., n_frames=30, is_circle=True)`。
圆半径取当前服务器 Inpaint360GS 对照实验的 `object_inpaint` 配置；训练相机排序、
LLFF hold=8 划分、首训练相机内参及实际原生虚拟图尺寸均采用官方规则。
每个场景的 `tracker/official_camera_source.json` 记录来源文件、哈希与训练图列表。
圆半径采用官方保存的四位小数值，与官方 inpaint 重新生成相机的方式一致。

`MASK_DILATION=10`，`MASK_DILATION_SHAPE=square`：对应官方 PIL `MaxFilter(21)`。
连通域最小面积为 50；无合格连通域时保留最大连通域。
完全遮挡的帧可使用空 mask；整段轨迹都无可见目标或整帧 mask 会报错。
空 mask 帧不会伪造目标，DeAOT 在目标重新可见时重新建立参考帧。

其余设置继承当前报告登记的 PaintMesh 5000 实验：PGSR 渲染、原始外观与几何监督、
depth=true、normal=false、surface_init=none、局部旧高斯更新与远处冻结、仅剪枝。
已修正的删除对象几何复用，但虚拟图、跟踪、LaMa、初始化与训练重新生成。
每场景完整配置保存到本批次 `configs/<scene>.yaml`；例如 garden_toys 保留
洞内 RGB 权重 0.25，cone_yellow/toys 保留已有严格局部更新配置。
支持帧数随相机数量换算：这两个场景的 `min_views=18/180` 改为 `3/30`，
保持 10% 的支持比例；其余场景仍要求至少两个视角支持。
初始化 `max_points=null`，沿用用户取消 500,000 上限的要求。

统一训练 10000 步，5000 步保存中间快照；训练结束后评估两个模型并提取最终 mesh。
`RECURSIVE_GUIDE=false`。测试 GT 只用于评估。
本次仅对齐相机与膨胀设置，训练算法仍为 PaintMesh。

已有批次可断点续跑（锁防止同一批次重复启动）：

```bash
cd /home/member/wyc/gsagent
nohup /home/member/.conda/envs/paintmesh/bin/python -u \
  scripts/paintmesh/run_inpaint360_official_camera.py \
  --job logs/paintmesh_circle30_d10_gpu1_20261030_114053 \
  >> logs/paintmesh_circle30_d10_gpu1_20261030_114053/runner.log 2>&1 < /dev/null &
```

查看 `status.json` 中 `current_log` 对应的阶段日志。脚本依次使用 GPU 1，启动每个
阶段前检查可用显存至少 17500 MiB，不终止其他作业。

结果直接更新 `logs/paintmesh_full_r4_gpu1_20261025_105925/comparison.md` 中原来的
逐场景指标和均值。新结果不按优劣选择；未完成场景显示待完成，均值仅使用五列方法
共同完成的场景。原报告和注册记录保存在本批次 `previous_report/`。
旧 `qualitative_*` 图片对应旧 adaptive180 实验，不能当作本轮结果。
