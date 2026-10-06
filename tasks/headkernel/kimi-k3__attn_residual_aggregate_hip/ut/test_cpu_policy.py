"""Synthetic CPU policy tests; no GPU or existing fixture deserialization."""
import json
from pathlib import Path
import tempfile
import unittest
from cpu_policy import configure_cpu_threads

class Threads:
    def __init__(self,value):self.value=value;self.calls=[]
    def get_num_threads(self):return self.value
    def set_num_threads(self,value):self.calls.append(value);self.value=value
    def get_num_interop_threads(self):return 16

class ThreadPolicyTests(unittest.TestCase):
    def test_large_pool_is_capped_and_before_after_are_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            torch=Threads(96);result=configure_cpu_threads(torch,Path(root))
            self.assertEqual(torch.calls,[8]);self.assertEqual(result['intraop_before'],96)
            self.assertEqual(result['intraop_after'],8);self.assertFalse(result['checks_or_samples_removed'])
            self.assertEqual(json.loads((Path(root)/'cpu_thread_policy.json').read_text()),result)
    def test_smaller_existing_pool_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            torch=Threads(4);configure_cpu_threads(torch,Path(root));self.assertEqual(torch.calls,[4])

if __name__=='__main__':unittest.main()
