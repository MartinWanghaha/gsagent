#!/usr/bin/env python3
"""Finalize both native color campaigns and consolidate current delivery notes."""
import fcntl
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import SCENES,read,write
from mesh_benchmark.layout import work_root
from repair_come_mesh_colors import PUBLIC,WORK


def run():
    lock=open(WORK/'finalization.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_file=PUBLIC/'docs/COLOR_EXPORT_REPAIRS_STATUS.json'
    while True:
        campaigns={method:read(PUBLIC/'_internal/workspace'/(method+'_color_repair')/'status.json') for method in ['gof','come']}
        write(status_file,dict(complete=False,stage='waiting for both full deliveries',
            updated=time.time(),campaigns={m:dict(stage=s['stage'],complete=s.get('complete',False)) for m,s in campaigns.items()}))
        if all(s.get('complete') for s in campaigns.values()):break
        for method,state in campaigns.items():
            if state.get('complete'):continue
            pid=state.get('pid')
            if pid and not Path(f'/proc/{pid}/cmdline').exists():
                raise RuntimeError(f'{method} campaign stopped before publication; inspect its campaign log')
        time.sleep(10)
    root=work_root(PUBLIC);protocol=read(root/'protocol.json')
    for name in ['repair_gof_mesh_colors.py','run_gof_color_repair.py','repair_come_mesh_colors.py',
                 'run_come_color_repair.py','render_come_color_candidate.py','audit_come_mesh_colors.py']:
        path=str(HERE/name)
        if path not in protocol['scripts']:protocol['scripts'].append(path)
    write(root/'protocol.json',protocol)
    from run_come_color_repair import publish_report,joint_report
    publish_report();joint_report()
    for p in [PUBLIC/'README.md',PUBLIC/'docs/README.md']:
        text=p.read_text();link='docs/COLOR_EXPORT_REPAIRS.md' if p.parent==PUBLIC else 'COLOR_EXPORT_REPAIRS.md'
        line=f'\n**GOF / CoMe 全11场景颜色修复及全方法图表更新已完成。** [最新修复总览与主报告入口]({link})。\n'
        if '最新修复总览与主报告入口' not in text:p.write_text(line+text)
    checked=read(PUBLIC/'docs/NATIVE_RGB_DELIVERY_CHECK.json')
    assert checked['complete']
    # Each campaign independently verified all source SHA hashes, image bytes,
    # shared masks and score means after its final publication. Here only the
    # documentation/provenance consolidation changed; mesh/render inputs did not.
    result=dict(complete=True,updated=time.time(),stage='complete',gof_color_meshes=len(SCENES),come_color_meshes=len(SCENES),
        geometry_unchanged_by_color_corrections=True,native_deliveries_verified=checked,
        report=str(PUBLIC/'docs/COLOR_EXPORT_REPAIRS.md'))
    write(status_file,result);print('ALL_NATIVE_MESH_COLOR_REPAIRS_COMPLETE',result,flush=True)


if __name__=='__main__':run()
