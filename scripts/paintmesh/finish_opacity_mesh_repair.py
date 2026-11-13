#!/usr/bin/env python3
"""Finalize the background repair after its scheduler exits; never publish stale scores."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write
from mesh_benchmark.layout import work_root
from repair_opacity_meshes import WORK


def finish(pid=None):
    public=BASE/'mesh_comparison_whole';root=work_root(public)
    lock=open(WORK/'delivery.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if pid:
        print('WAIT_REPAIR_SCHEDULER',pid,flush=True)
        while True:
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'run_opacity_mesh_repair.py' not in cmd:break
            time.sleep(10)
    for attempt in range(2):
        if read(WORK/'status.json')['stage']=='complete':break
        print('RESUME_INCOMPLETE_REPAIR',attempt+1,flush=True)
        subprocess.run([sys.executable,'-u',str(HERE/'run_opacity_mesh_repair.py')],check=attempt==1)
    assert read(WORK/'status.json')['stage']=='complete','Repair did not complete; reports are not claimed current'
    write(WORK/'delivery_status.json',dict(stage='source and native depth audit',complete=False,updated=time.time()))
    from audit_mesh_comparison import structure,depths,provenance
    structure(public);depths(public);provenance(public)
    from publish_opacity_mesh_repair import publish_support_report
    records=publish_support_report(public)
    from verify_native_mesh_delivery import verify
    verify(public)
    # Keep the older color check explicit about its stage: colors and vertex
    # attributes are exact, while CoMe/GOF evaluated faces have been validated.
    path=root/'colored_mesh_delivery_check.json';old=read(path) if path.exists() else {}
    old.update(complete=True,all_geometry_unchanged=False,geometry_cache_identities_preserved=False,
        all_vertex_coordinates_and_native_colors_preserved=True,geometry_postprocessing='CoMe/GOF source Gaussian support validation',
        current_geometry_cache_matches_verified=True,repair_report=str(public/'docs/OPACITY_MESH_REPAIR_RESULTS.json'))
    for entry in old.get('meshes',[]):
        if entry['method']=='come':
            match=next(r for r in records if r['method']=='come' and r['scene']==entry['scene'])
            entry.update(faces=match['repair']['faces'],colored_path=match['repair']['path'],
                geometry_unchanged=False,vertex_coordinates_and_colors_exact=True)
    write(path,old)
    p=public/'docs/COLORED_MESHES.md';text=p.read_text()
    text=text.replace('所有顶点坐标、三角形及其余原属性逐项核对，几何不变。',
        '着色阶段保留几何；本轮 CoMe / GOF 网格另经源高斯支撑校验，原坐标与颜色不变，无支撑的长插值面被剔除，详见 [修复协议](OPACITY_MESH_REPAIR.md)。')
    text=text.replace('当前 CD/F1 无需因颜色改变而重算。','当前 CD/F1 已针对支撑校验后的网格重新计算。')
    p.write_text(text)
    p=public/'README.md';text=p.read_text()
    banner='**CoMe / GOF 网格支撑校验修复正在执行，全场景指标与图片将覆盖更新。** [当前进度](docs/OPACITY_MESH_REPAIR_STATUS.md)。\n\n'
    text=text.replace(banner,'**CoMe / GOF 全场景网格修复及重评估已完成。** 当前网格采用原生提取加源高斯支撑校验，原始文件保留。详见 [修复规则、同像素指标与前后图片](docs/OPACITY_MESH_REPAIR.md)。\n\n')
    text=text.replace('CoMe / Wrapping 的 22 份彩色 PLY 已核验，原几何不变。','CoMe / Wrapping 的 22 份彩色 PLY 已核验；CoMe / GOF 当前提取另经支撑校验，原坐标与原生颜色不变。')
    text=text.replace('CoMe truck的原生mesh质量异常已保留实际分数，见 [质量观察](docs/QUALITY_NOTES.md)。','CoMe truck 的原生跨空巨型遮挡面已处理，所有改善与覆盖变化见 [修复对比](docs/OPACITY_MESH_REPAIR.md)。')
    p.write_text(text)
    p=public/'docs/README.md';text=p.read_text()
    if 'OPACITY_MESH_REPAIR.md' not in text:text+='\n- [CoMe / GOF 支撑校验修复与同像素对比](OPACITY_MESH_REPAIR.md)\n'
    p.write_text(text)
    p=public/'docs/SYNC_LATEST.md';text=p.read_text()
    text+='\nCoMe / GOF 本轮22份原生网格支撑校验及110份指标重评估已完成，132组子图已覆盖；原生提取、后处理规则与同像素改善见 [修复结果](OPACITY_MESH_REPAIR.md)。\n';p.write_text(text)
    write(WORK/'delivery_status.json',dict(complete=True,stage='complete',updated=time.time(),
        support_validated_meshes=len(records),metric_groups=110,subplot_groups=132,
        report=str(public/'docs/OPACITY_MESH_REPAIR.md')))
    print('OPACITY_MESH_REPAIR_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int);a=p.parse_args();finish(a.after_pid)
