"""Guard policy tests: spare VRAM alone must never admit a competing job."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_hard_reference_all import gpu_decision, Worker


class GuardTests(unittest.TestCase):
    def snapshot(self, free=24000, util=0, processes=()):
        return dict(free_mib=free, used_mib=24576-free, utilization=util,
                    processes=list(processes))

    def test_busy_foreign_process_even_with_spare_vram(self):
        s=self.snapshot(processes=[dict(pid=222,memory_mib=100)])
        self.assertEqual(gpu_decision(s), 'foreign_gpu_process')

    def test_idle_card_required(self):
        self.assertIsNone(gpu_decision(self.snapshot()))
        self.assertEqual(gpu_decision(self.snapshot(free=21000)), 'insufficient_free_memory')
        self.assertEqual(gpu_decision(self.snapshot(util=25)), 'gpu_busy')

    def test_our_running_group_can_use_card(self):
        s=self.snapshot(free=10000,util=99,processes=[dict(pid=111,memory_mib=14000)])
        self.assertIsNone(gpu_decision(s,{111},running=True))
        s['processes'].append(dict(pid=222,memory_mib=100))
        self.assertEqual(gpu_decision(s,{111},running=True),'foreign_gpu_process')

    def test_memory_guards_before_capacity(self):
        s=self.snapshot(free=5000,processes=[dict(pid=111,memory_mib=18100)])
        self.assertEqual(gpu_decision(s,{111},running=True),'memory_risk')
        s=self.snapshot(free=3500,processes=[dict(pid=111,memory_mib=15000)])
        self.assertEqual(gpu_decision(s,{111},running=True),'memory_risk')

    def test_stop_signals_only_created_child_group(self):
        w=Worker.__new__(Worker);w.child=Mock(pid=111)
        w.child.poll.return_value=None
        with patch('run_hard_reference_all.os.killpg') as kill:
            w.stop_child()
            self.assertEqual(kill.call_count,1)
            self.assertEqual(kill.call_args.args[0],111)


if __name__=='__main__':unittest.main()
