"""Live report for exact original Gaussian reuse and independent generated-mesh checks."""
from pathlib import Path
from .io import read_json
from .evidence import SELECTION


def write_status(root,queue=None):
    root=Path(root);rows=[];gaussian_ready=0;mesh_ready=0
    for scene,group in SELECTION.items():
        folder=root/scene;status=read_json(folder/'status.json') if (folder/'status.json').exists() else dict(state='pending')
        path=folder/'composition/world/gaussians_manifest.json'
        gs=read_json(path) if path.exists() else {}
        ready=bool(gs.get('complete') and gs.get('training') is False and (path.parent/'composed_gaussians.ply').exists())
        gaussian_ready+=int(ready)
        scene_path=folder/'composition/scene.json';graph=read_json(scene_path) if scene_path.exists() else {}
        mesh=bool(status.get('state')=='complete' and graph.get('schema_version')==2 and graph.get('components',{}).get('mesh',{}).get('status')=='complete')
        mesh_ready+=int(mesh)
        counts='; '.join(f'ID {key}: {value}' for key,value in gs.get('object_counts',{}).items())
        detail=status.get('error','; '.join(status.get('reasons',[])))
        phase=status.get('stage',status['state'])
        rows.append(f"| {scene} | {group} | {phase} | {'YES' if ready else 'NO'} | {counts or '-'} | {'YES' if mesh else 'NO'} | {detail} |")
    text=['# 原始物体高斯 + 生成 mesh：重生成状态','',
          '**3DGS = inpaint 背景 + 实际删除的原始高斯；不重新训练。mesh = inpaint 背景 + MV-SAM3D 物体 mesh。**','',
          f'新 3DGS 已输出：{gaussian_ready}/11；mesh 位姿通过当前阈值：{mesh_ready}/11。','']
    if (root/'active_jobs.json').exists():
        jobs=read_json(root/'active_jobs.json');live=[]
        for job in jobs['jobs']:
            try:cmd=Path(f'/proc/{job["pid"]}/cmdline').read_bytes()
            except FileNotFoundError:continue
            if any(s in cmd for s in [b'run_recompose.py',b'resume_recompose_outputs.py',b'review_recomposition.py',b'regenerate_native_recomposition.py']):
                live.append(f"{job['task']} (PID {job['pid']})")
        text+=['当前任务：'+('；'.join(live) if live else '后台任务已结束，查看逐场景状态。'),'',f"日志：`{jobs['logs']}`。",'']
    text+=['| 场景 | 背景来源 | 当前阶段 | 新 3DGS 已输出 | 恢复的原始物体点数 | mesh 通过检查 | 问题 |',
           '|---|---|---|---|---|---|---|',*rows,'',
           '3DGS：`<scene>/composition/world/composed_gaussians.ply`。mesh：同目录 `mesh_scene.gltf`、独立物体 mesh / GLB。',
           '可移动主场景：`<scene>/composition/scene.json`，schema_version=2。mesh 的 needs_review 不影响原始高斯直接复用的完成状态。']
    (root/'STATUS.md').write_text('\n'.join(text)+'\n')
