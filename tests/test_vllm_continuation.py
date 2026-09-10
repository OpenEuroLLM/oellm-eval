import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]/"containers/continuation_scoring"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load("vllm_continuation_engine")
adapter = load("vllm_continuation_adapter")


def params(start=None):
    return NS(extra_args=None if start is None else {engine.KEY:start}, prompt_logprobs=1, flat_logprobs=False)


class ContinuationTests(unittest.TestCase):
    def test_disabled_or_missing_engine_keeps_legacy_path(self):
        with patch.object(adapter, "import_module") as imports:
            self.assertFalse(adapter.available("false"))
            imports.assert_not_called()
            imports.side_effect = ModuleNotFoundError(name="vllm.oellm_continuation")
            self.assertFalse(adapter.available(True))
        with self.assertRaises(ValueError):
            adapter.available("maybe")

    def test_partial_engine_or_wrong_runner_fails_closed(self):
        with patch.object(adapter, "import_module", side_effect=[NS(OELLM_CONTINUATION_API=1), NS()]):
            with self.assertRaises(RuntimeError):
                adapter.available(True)
        with patch.object(adapter, "import_module", return_value=NS(OELLM_CONTINUATION_API=1)):
            with patch.dict(adapter.os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
                with self.assertRaises(ValueError):
                    adapter.available(True)

    def test_every_target_survives_random_cache_and_chunk_boundaries(self):
        rng = random.Random(43)
        for _ in range(500):
            boundary = rng.randrange(1,200)
            length = boundary + rng.randrange(1,100)
            computed = rng.randrange(boundary)  # safe cache can only cover context
            recovered = []
            while computed < length:
                scheduled = min(rng.randrange(1,65), length-computed)
                positions, ranges = engine.scoring_ranges([0,scheduled], [computed], [scheduled], [length], [boundary], [True])
                recovered.extend(computed + row + 1 for row in positions)
                self.assertEqual(ranges, [(0,len(positions))])
                computed += scheduled
            self.assertEqual(recovered, list(range(boundary,length)))

    def test_mixed_requests_exclude_generation_and_include_standard_logprobs(self):
        pos, ranges = engine.scoring_ranges([0,4,9,12], [6,0,0], [4,5,3], [12,5,3], [8,1,1], [True,True,False])
        self.assertEqual(pos,[1,2,3,4,5,6,7])
        self.assertEqual(ranges,[(0,3),(3,7),(7,7)])

    def test_empty_prefix_chunk_emits_no_scores(self):
        self.assertEqual(engine.scoring_ranges([0,32],[0],[32],[100],[80],[True]), ([],[(0,0)]))

    def test_cache_stops_before_first_target_prediction(self):
        for start in (1,15,16,17,31,32,33,127):
            request = NS(sampling_params=params(start),num_tokens=start+83)
            self.assertEqual(engine.cache_limit(request,request.num_tokens-1),start-1)
        self.assertEqual(engine.cache_limit(NS(sampling_params=params(),num_tokens=90),89),89)

    def test_prefix_is_none_and_invalid_boundaries_fail(self):
        self.assertEqual(engine.initial_logprobs(params(3),[1,2,3,4],lambda _: [None]), [None]*3)
        for invalid in (0,-1,True,2.0,"2"):
            with self.assertRaises(ValueError):
                engine.start_position(params(invalid),10)
        with self.assertRaises(ValueError):
            engine.start_position(params(10),10)
        p=params(2)
        p.flat_logprobs=True
        with self.assertRaises(ValueError):
            engine.start_position(p,10)

    def test_groups_stay_on_same_gpu_and_results_restore(self):
        requests=[[1,2,10],[9,3],[1,2,11],[9,4],[8,1],[8,2]]
        config=[params(2),params(1),params(2),params(1),params(1),params(1)]
        shards=adapter.partition(requests,config,4)
        self.assertEqual(len(shards),3)
        for group in ([0,2],[1,3],[4,5]):
            self.assertTrue(any(set(group)<=set(shard) for shard in shards))
        outputs=[[f"answer{i}" for i in shard] for shard in shards]
        self.assertEqual(adapter.gather(shards,outputs,6),[f"answer{i}" for i in range(6)])

    def test_standard_generation_keeps_round_robin(self):
        self.assertEqual(adapter.partition([[i] for i in range(7)],[params()]*7,4),[[0,4],[1,5],[2,6],[3]])

    def test_first_endings_precede_alternatives_to_warm_prefixes(self):
        requests=[[q,token] for q in range(4) for token in (10,11)]
        shards=adapter.partition(requests,[params(1)]*8,1,wave_size=2)
        self.assertEqual(shards,[[0,2,1,3,4,6,5,7]])

    def test_gather_fails_on_missing_or_duplicated_results(self):
        with self.assertRaises(ValueError):
            adapter.gather([[0],[0]],[["a"],["b"]],2)
        with self.assertRaises(ValueError):
            adapter.gather([[0,1]],[["a"]],2)


if __name__ == "__main__":
    unittest.main()
