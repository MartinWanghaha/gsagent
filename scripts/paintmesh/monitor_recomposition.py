#!/usr/bin/env python3
"""Refresh the small live report while registered background jobs are alive."""
from pathlib import Path
import time
from recompose.io import read_json
from recompose.report import write_status


def main():
    root=Path(__file__).resolve().parents[2]/'output/paintmesh/Inpaint360GS/recomposition'
    while True:
        jobs=read_json(root/'active_jobs.json');alive=False
        for job in jobs['jobs']:
            try:cmd=Path(f'/proc/{job["pid"]}/cmdline').read_bytes()
            except FileNotFoundError:continue
            alive|=any(s in cmd for s in [b'run_recompose.py',b'resume_recompose_outputs.py',b'review_recomposition.py'])
        queue=read_json(root/'queue_status.json') if (root/'queue_status.json').exists() else None
        write_status(root,queue)
        if not alive:return
        time.sleep(20)


if __name__=='__main__':main()
