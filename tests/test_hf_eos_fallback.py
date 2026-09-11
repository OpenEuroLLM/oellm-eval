"""Execute the fallback shipped in the runtime patch; preserve explicit choices."""
from pathlib import Path
from types import SimpleNamespace
import unittest

patch=(Path(__file__).resolve().parents[1]/'patches/harness-hf-eos-fallback.patch').read_text()
added=[line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
body='\n'.join(line[4:] for line in added)
namespace={}
exec('def apply(self, generation_kwargs):\n'+body+'\n    return generation_kwargs\n',namespace)


class EOSFallback(unittest.TestCase):
    def apply(self,configured=None,tokenizer=2,**kwargs):
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=configured))
        return namespace['apply'](SimpleNamespace(model=model,eot_token_id=tokenizer),kwargs)

    def test_missing_export_eos_uses_tokenizer(self):
        self.assertEqual(self.apply(),{'eos_token_id':2})

    def test_configured_list_preserved(self):
        self.assertEqual(self.apply(configured=[2,3]),{})

    def test_explicit_override_preserved(self):
        self.assertEqual(self.apply(eos_token_id=7),{'eos_token_id':7})

    def test_explicit_none_preserved(self):
        self.assertEqual(self.apply(eos_token_id=None),{'eos_token_id':None})

    def test_missing_tokenizer_eos_is_not_invented(self):
        self.assertEqual(self.apply(tokenizer=None),{})


if __name__=='__main__':unittest.main()
