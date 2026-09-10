"""Regression checks execute the materialization code shipped in the patch."""
from pathlib import Path
import unittest

patch=(Path(__file__).resolve().parents[1]/'patches/harness-greedy-default.patch').read_text()
added=[line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
body='\n'.join(line[4:] for line in added)
namespace={}
exec('def materialize(value):\n    _gen_kwargs = dict(value)\n'+body+'\n    return _gen_kwargs\n',namespace)
materialize=namespace['materialize']

class GreedyDefault(unittest.TestCase):
    def test_inferred_greedy_materializes_zero(self):
        self.assertEqual(materialize({'do_sample':False,'until':['\n']}),{'temperature':0.0,'until':['\n']})
    def test_explicit_greedy_overrides_sampling_temperature(self):
        self.assertEqual(materialize({'do_sample':False,'temperature':.7}),{'temperature':0.0})
    def test_explicit_sampling_is_preserved(self):
        self.assertEqual(materialize({'do_sample':True,'temperature':.7}),{'temperature':.7})
    def test_explicit_sampling_can_use_vllm_default(self):
        self.assertEqual(materialize({'do_sample':True}),{})

if __name__=='__main__':unittest.main()
