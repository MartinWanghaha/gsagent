#!/usr/bin/env python3
"""Resume scoring/publication from already completed native CoMe renders."""
import fcntl
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,STATES,read,write
from mesh_benchmark.layout import work_root
from recompose.io import fingerprint
from repair_come_mesh_colors import PUBLIC,WORK
from run_come_color_repair import status,publish_report,joint_report


def run():
    lock=open(WORK/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=work_root(PUBLIC);locks=[]
    for name in ['native_rgb_campaign.lock','campaign.lock']:
        f=open(root/name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    # Rendering is complete; compare its recorded mesh SHA with the published
    # marker. The final verifier rehashes the actual files independently.
    for scene in SCENES:
        marker=read(BASE/'come'/scene/'mesh_complete.json')
        native=read(root/'native_renderers/come'/scene/'complete.json')
        assert native['complete'] and not native['pilot']
        assert native['request']['mesh']==dict(path=str(Path(marker['path']).resolve()),
            size_bytes=Path(marker['path']).stat().st_size,sha256=marker['sha256'])
        assert native['request']['adapter']==fingerprint(HERE/'render_native_colored_meshes.py')
    from evaluate_native_mesh_rgb import schedule,status as native_status,reuse_unchanged_score
    tasks={'come/'+s+'/'+state:dict(status='pending') for s in SCENES for state in STATES}
    for key,row in tasks.items():
        method,scene,state=key.split('/')
        if reuse_unchanged_score(root,method,scene,state):row['status']='complete'
    status('scoring corrected native CoMe renders',tasks)
    schedule(root,tasks,'score')
    from mesh_benchmark.report import update
    update(root)
    for script,args in [('plot_mesh_method_comparison.py',['--root',str(root)]),
                        ('organize_mesh_subplots.py',['--figures',str(PUBLIC/'figures_all_methods')])]:
        with (WORK/(script+'.log')).open('w') as f:
            subprocess.run([sys.executable,str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    native_status(root,'complete',tasks)
    status('verifying scores and publication',tasks)
    publish_report()
    from verify_native_mesh_delivery import verify
    verify(PUBLIC);joint_report()
    status('complete',{s:dict(status='complete') for s in SCENES})
    print('COME_COLOR_REPAIR_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':run()
