"""Verify scheduling exclusivity, checkpoint affinity and isolated status writes."""
import concurrent.futures
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
from run_muti_plan_parallel import Worker, claim_task, choose_task, finish_task


def test_resume_affinity_does_not_block_other_gpu():
    tasks = [dict(key="resume", state="pending", preferred_gpu=1),
             dict(key="done", state="complete"), dict(key="fresh", state="pending")]
    assert choose_task(tasks, 0)["key"] == "fresh"
    assert choose_task(tasks, 1)["key"] == "resume"
    tasks[0]["state"] = tasks[2]["state"] = "running"
    assert choose_task(tasks, 0) is None and choose_task(tasks, 1) is None


def claim_all(root, gpu):
    keys = []
    while (task := claim_task(root, gpu)) is not None:
        keys.append(task["key"])
    return keys


def test_parallel_claims_are_exactly_once(tmp_path):
    (tmp_path / "parallel").mkdir()
    tasks = [dict(key=str(i), state="pending") for i in range(39)]
    (tmp_path / "parallel/queue.json").write_text(json.dumps(dict(tasks=tasks)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim_all, tmp_path, g) for g in (0, 1)]
        keys = sum([f.result() for f in futures], [])
    assert len(keys) == 39 and set(keys) == {str(i) for i in range(39)}
    assert all(t["state"] == "running" for t in json.loads((tmp_path / "parallel/queue.json").read_text())["tasks"])


def test_task_cannot_be_completed_by_wrong_gpu(tmp_path):
    (tmp_path / "parallel").mkdir()
    (tmp_path / "parallel/queue.json").write_text(json.dumps(dict(tasks=[dict(key="x", state="pending")])))
    claim_task(tmp_path, 1)
    with pytest.raises(ValueError, match="ownership"):
        finish_task(tmp_path, "x", 0, "complete")
    finish_task(tmp_path, "x", 1, "complete")
    assert claim_task(tmp_path, 1) is None


def test_worker_status_does_not_overwrite_shared_status(tmp_path):
    shared = tmp_path / "status.json"
    shared.write_text('{"state":"controller-owned"}')
    worker = Worker.__new__(Worker)
    worker.state = {"gpu": 0}
    worker.state_path = tmp_path / "parallel/workers/gpu0.json"
    worker.save(state="running", child_pid=123)
    assert json.loads(shared.read_text()) == {"state": "controller-owned"}
    assert json.loads(worker.state_path.read_text())["child_pid"] == 123
