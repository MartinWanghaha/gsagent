#!/usr/bin/env python3
"""Organize a completed whole-mesh benchmark without modifying numeric results."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from mesh_benchmark.layout import MARKER,work_root,publish_reports

METHODS=['2dgs','come','gof','gaussian_wrapping','paintmesh']
STATES=['background','recomposition']


def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def checksum(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def organize(root):
    root=root.resolve();internal=root/'_internal';work=internal/'workspace'
    journal=internal/'organization.json'
    if (root/MARKER).exists():return work_root(root)
    work.mkdir(parents=True,exist_ok=True)
    record=read(journal) if journal.exists() else dict(started=datetime.now().isoformat(),moves=[],metrics={})
    if not record['metrics']:
        for state in STATES:
            for p in (root/state).glob('*/*/metrics.json'):
                record['metrics'][str(p.relative_to(root))]=checksum(p)
        record['source_identities']={}
        for p in list((root/'meshes').glob('*/background.ply'))+list((root/'recomposition').glob('*/composition/scene.json')):
            st=p.stat();record['source_identities'][str(p.relative_to(root))]=[st.st_ino,st.st_size,st.st_mtime_ns]
    write(journal,record)
    def move(source,dest):
        if source==dest:return
        if source.exists() or source.is_symlink():
            if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
            dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
            record['moves'].append([str(source),str(dest)]);write(journal,record)
    def link(target,dest):
        if not dest.exists() and not dest.is_symlink():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(os.path.relpath(target,dest.parent),target_is_directory=target.is_dir())
    docs=root/'docs';docs.mkdir(exist_ok=True)
    results=root/'results';results.mkdir(exist_ok=True)
    original_docs={}
    for p in list(root.glob('*.md')):
        if p.name=='README.md':
            move(p,work/'README.before_organization.md')
        else:
            original_docs[p.name]=p.read_text();move(p,docs/p.name)
    for name in ['cache','common','models','logs']:
        move(root/name,internal/name);link(internal/name,work/name)
    move(root/'background',results/'background');link(results/'background',work/'background')
    (results/'recomposition').mkdir(exist_ok=True);(work/'recomposition').mkdir(exist_ok=True)
    for name in [*METHODS,'figures','report.md']:
        move(root/'recomposition'/name,results/'recomposition'/name)
        link(results/'recomposition'/name,work/'recomposition'/name)
    scenes=read(root/'protocol.json')['scenes'] if (root/'protocol.json').exists() else read(work/'protocol.json')['scenes']
    for scene in scenes:link(root/'recomposition'/scene,work/'recomposition'/scene)
    for name in ['meshes','figures_all_methods']:link(root/name,work/name)
    for p in list(root.iterdir()):
        if p.is_file() and p.suffix in ['.json','.log','.pid','.lock']:
            if p.name==MARKER:continue
            move(p,work/p.name)
    # Human documents remain in docs; machine configuration has one canonical
    # writable copy in workspace and a view in docs, avoiding stale file aliases.
    for name in ['protocol.json','whole_scene_plan.json','pilot_parameters.json','DELIVERY_CHECK.json']:
        link(work/name,docs/name)
    for p in docs.glob('*.md'):link(p,work/p.name)
    config=dict(version=1,public_root=str(root),work_root=str(work),
                note='Directory-only working aliases; source mesh/scene identities and metric bytes retained')
    write(root/MARKER,config);link(root/MARKER,work/MARKER)
    # Rebase moved document links using the explicit move journal. Source mesh
    # identities are untouched; only navigation text is edited.
    moves=sorted([(Path(a),Path(b)) for a,b in record['moves']],key=lambda pair:-len(str(pair[0])))
    def relocated(p):
        for old,new in moves:
            try:return new/p.relative_to(old)
            except ValueError:pass
        return p
    for name,text in original_docs.items():
        def replace(match):
            target=match.group(2)
            if target.startswith(('#','http:','https:','mailto:')):return match.group(0)
            bare,sep,fragment=target.strip('<>').partition('#')
            absolute=Path(os.path.abspath(root/bare));actual=relocated(absolute)
            value=os.path.relpath(actual,docs)
            if sep:value+='#'+fragment
            if ' ' in value:value='<'+value+'>'
            return match.group(1)+value+')'
        text=re.sub(r'(!?\[[^\]]*\]\()([^\n)]+)\)',replace,text)
        (docs/name).write_text(text)
    figures=root/'figures_all_methods'
    for p in list(figures.iterdir()):
        if p.is_file():
            if p.name.startswith('all_scenes_') or p.name.endswith('_all_scenes.pdf'):move(p,figures/'overview'/p.name)
            elif p.suffix=='.json':move(p,figures/'metadata'/p.name)
            elif p.suffix in ['.pid','.lock']:move(p,figures/'.runtime'/p.name)
    record.update(organized=True,finished=datetime.now().isoformat());write(journal,record)
    from mesh_benchmark.layout import document_links
    for state in STATES:
        source=work/state/'report.md'
        source.write_text(document_links(source.read_text(),source,source.resolve()))
    publish_reports(work)
    indexes(root,scenes)
    return work


def indexes(root,scenes):
    (root/'README.md').write_text('''# 全场景 Mesh 对比结果

11个场景，四个基线与PaintMesh纯背景/组合两种模式。176个评估子任务、110份模式指标；独立子图132组。

| 内容 | 入口 | 目录 |
|---|---|---|
| 纯背景：几何与渲染指标 | [报告](reports/background.md) | `reports/` |
| 背景＋MV-SAM3D：几何与渲染指标 | [报告](reports/recomposition.md) | `reports/` |
| 全部方法总图、逐场景详图、独立子图 | [图片索引](figures_all_methods/README.md) | `figures_all_methods/` |
| 每个方法/场景的指标、渲染、评分mask | [数据索引](results/README.md) | `results/<模式>/<方法>/<场景>/` |
| 我们的完整可观测背景mesh | [背景网格](meshes/README.md) | `meshes/<场景>/` |
| 加入MV-SAM3D的组合网格资产 | [组合网格](recomposition/README.md) | `recomposition/<场景>/composition/` |
| 配置、协议、覆盖率、质量观察 | [说明索引](docs/README.md) | `docs/` |
| 缓存、相机/参考点、深度融合帧、日志、运行记录 | [内部目录说明](_internal/README.md) | `_internal/` |

常用图片： [RGB总图](figures_all_methods/overview/all_scenes_rgb.jpg) · [几何总图](figures_all_methods/overview/all_scenes_geometry.jpg) · [各方法子图](figures_all_methods/subplots/README.md)。PNG/JPG/PDF总图统一在 `figures_all_methods/overview/`。

几何指标含CD与Precision/Recall/F1。主阈值为相机半径1%，参考为COLMAP稀疏点云，表示已观测表面一致性。图像评分只取共同有效像素；纯背景排除目标mask内部。具体口径见 [评估协议](docs/PROTOCOL.zh-CN.md)。

CoMe truck的原生mesh质量异常已保留实际分数，见 [质量观察](docs/QUALITY_NOTES.md)。

目录整理只搬移文件及更新导航；指标数值和源网格保持原样。复用评估、绘图脚本仍可将本目录作为 `--output` / `--root` 传入，脚本会自动解析 `_internal/workspace` 的兼容目录映射。
''')
    (root/'docs/README.md').write_text('''# 配置与说明

- [评估协议](PROTOCOL.zh-CN.md) · [机器配置](protocol.json)
- [全背景提取方案](WHOLE_SCENE_PLAN.zh-CN.md) · [逐场景参数](whole_scene_plan.json)
- [实际融合参数/内存调整](FUSION_MEMORY_ADJUSTMENTS.md)
- [背景覆盖率对照](COVERAGE_COMPARISON.md)
- [各方法mesh路径索引](MESH_FILES.md)
- [质量观察](QUALITY_NOTES.md) · [任务完成状态](WHOLE_STATUS.md)
- [先前交付核验](DELIVERY_CHECK.json)
- [目录整理核验](../_internal/organization_check.json)

原文中的程序相对路径对应 `_internal/workspace/`；常用文件按根目录README中的实际入口访问。
''')
    (root/'results/README.md').write_text('''# 逐方法、逐场景评估数据

- `background/<method>/<scene>/`：纯背景模式；目标mask内不评分。
- `recomposition/<method>/<scene>/`：原始物体存在状态/组合模式。
- `metrics.json`：逐视角和场景均值，含CD、F1、覆盖率与评分范围。
- `renders/`：真实mesh的统一颜色场渲染。
- `diagnostics/`：实际评分mask、覆盖mask、误差图。
- `color_field.pt`：冻结mesh几何的外观模型；`invocation.json`记录调用。

方法名：`2dgs`、`come`、`gof`、`gaussian_wrapping`、`paintmesh`。每种方法均包含11场景。两份汇总报告见 `../reports/`，完整逐方法子图见 `../figures_all_methods/subplots/`。
''')
    for folder,title,link in [('meshes','完整可观测背景mesh','background.ply'),('recomposition','背景＋MV-SAM3D组合mesh','composition/world/mesh_scene.gltf')]:
        text=[f'# {title}','','| 场景 | 主要文件 |','|---|---|']
        for s in scenes:text.append(f'| {s} | [打开]({s}/{link}) |')
        text+=['','逐场景评估数值和渲染统一在 `../results/`，这里存储网格/场景资产。']
        (root/folder/'README.md').write_text('\n'.join(text)+'\n')
    (root/'_internal/README.md').write_text('''# 内部数据与运行资料

| 目录/文件 | 内容 |
|---|---|
| `cache/` | 各方法实际mesh深度、表面采样、外观训练池、原生颜色渲染 |
| `common/` | 公共相机、GT、目标mask、共同评分域、稀疏点参考 |
| `models/` | 全背景提取参数、缓存RGB/深度帧、提取完成凭据 |
| `logs/` | 提取、评估、绘图和核验日志 |
| `workspace/` | 程序兼容目录映射，以及protocol/results/status、PID/锁、恢复记录 |
| `organization.json` | 逐项搬移日志、原指标SHA256、源资产身份 |
| `organization_check.json` | 整理后的文件与数值核验 |

`workspace`中的目录链接供原脚本使用，不是重复数据。保留内部缓存可继续评估和绘图，无需重新训练或重复复制大型网格。
''')


def check(root):
    root=root.resolve();work=work_root(root);journal=read(root/'_internal/organization.json')
    for old,expected in journal['metrics'].items():
        state,relative=old.split('/',1);assert checksum(work/state/relative)==expected,old
    for relative,identity in journal['source_identities'].items():
        s=(root/relative).stat();assert [s.st_ino,s.st_size,s.st_mtime_ns]==identity,relative
    files=[]
    for state in STATES:
        for p in (work/state).glob('*/*/metrics.json'):
            d=read(p);assert d['complete'] and d['scoring_protocol_version']==2
            from mesh_benchmark.core import identity as file_identity
            for specification in d['request']['geometry']['specs']:
                assert file_identity(specification['identity']['path'])==specification['identity']
            files.append(str(p))
    assert len(files)==110
    for scene in read(work/'protocol.json')['scenes']:
        common=read(work/'common'/scene/'complete.json')
        from mesh_benchmark.core import digest,identity as file_identity
        assert digest(common['request'])==common['request_id']
        for identity in [common['request']['split'],common['request']['selection'],*common['request']['masks']]:
            assert file_identity(identity['path'])==identity
    result=dict(success=True,checked_at=datetime.now().isoformat(),metric_files=len(files),
                unchanged_metric_sha256=len(journal['metrics']),unchanged_source_identities=len(journal['source_identities']),
                source_assets_retrained=False,work_root=str(work))
    write(root/'_internal/organization_check.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if not a.check:organize(a.root)
    check(a.root)
