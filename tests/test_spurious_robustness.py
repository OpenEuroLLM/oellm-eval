"""Tests for the spurious-robustness contrib suite.

The worst-group cases carry hand-computed expected values: worst-group accuracy
is a minimum, so an off-by-one in group assignment produces a plausible-looking
number rather than an error.
"""

from __future__ import annotations

import pytest

from oellm.contrib.spurious_robustness import datasets as ds
from oellm.contrib.spurious_robustness import prompts
from oellm.contrib.spurious_robustness.adapter import OpenClipAdapter
from oellm.contrib.spurious_robustness.metrics import (
    AverageAccuracy,
    WorstGroupAccuracy,
)
from oellm.contrib.spurious_robustness.zeroshot import group_metrics


def _records(spec):
    """(group, n_correct, n_total) triples -> sample records."""
    out = []
    for group, n_correct, n_total in spec:
        out += [{"correct": True, "group": group}] * n_correct
        out += [{"correct": False, "group": group}] * (n_total - n_correct)
    return out


class TestWorstGroupAccuracy:
    def test_hand_computed_minimum(self):
        # blonde_male 1/4 = 0.25 is the minimum; blonde_female 3/4 = 0.75,
        # non-blonde_male 8/10 = 0.8, non-blonde_female 9/10 = 0.9.
        samples = _records(
            [
                ("blonde_male", 1, 4),
                ("blonde_female", 3, 4),
                ("non-blonde_male", 8, 10),
                ("non-blonde_female", 9, 10),
            ]
        )
        assert WorstGroupAccuracy().compute(samples) == pytest.approx(0.25)

    def test_average_is_over_samples_not_groups(self):
        """Group means and the overall mean differ under imbalance."""
        # 1/4 + 9/10 correct = 10 of 14 samples = 0.714..., whereas the mean of
        # the two group accuracies would be (0.25 + 0.9) / 2 = 0.575.
        samples = _records([("rare", 1, 4), ("common", 9, 10)])
        assert AverageAccuracy().compute(samples) == pytest.approx(10 / 14)
        assert AverageAccuracy().compute(samples) != pytest.approx(0.575)

    def test_single_group_degenerates_to_accuracy(self):
        samples = _records([("all", 3, 4)])
        assert WorstGroupAccuracy().compute(samples) == pytest.approx(0.75)
        assert AverageAccuracy().compute(samples) == pytest.approx(0.75)

    def test_group_with_zero_samples_is_absent_not_zero(self):
        """An unobserved group has no accuracy, so it cannot be the minimum.

        Scoring it 0.0 would make every subsampled run report a worst-group of
        zero regardless of the model.
        """
        samples = _records([("present_a", 2, 2), ("present_b", 1, 2)])
        # "missing_group" contributes nothing and must not drag the result to 0.
        assert WorstGroupAccuracy().compute(samples) == pytest.approx(0.5)

    def test_empty_and_malformed_input(self):
        assert WorstGroupAccuracy().compute([]) == 0.0
        assert AverageAccuracy().compute([]) == 0.0
        # Records missing a group, or not dicts at all, are dropped rather than
        # counted as wrong answers.
        assert WorstGroupAccuracy().compute([None, "junk", {"correct": True}]) == 0.0
        mixed = [{"correct": False, "group": "g"}, None, {"correct": True}]
        assert WorstGroupAccuracy().compute(mixed) == pytest.approx(0.0)

    def test_all_correct_and_all_wrong(self):
        assert WorstGroupAccuracy().compute(_records([("a", 2, 2), ("b", 3, 3)])) == 1.0
        assert WorstGroupAccuracy().compute(_records([("a", 0, 2), ("b", 3, 3)])) == 0.0


class TestGroupMetrics:
    def test_hand_computed_full_output(self):
        # a: 1/4 = 0.25 (worst), b: 3/4 = 0.75, c: 4/4 = 1.0 (best)
        # overall: 8 of 12 = 0.666...
        preds = [1, 0, 0, 0] + [1, 1, 1, 0] + [1, 1, 1, 1]
        labels = [1, 1, 1, 1] * 3
        groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4

        m = group_metrics(preds, labels, groups)
        assert m["worst_group_accuracy"] == pytest.approx(0.25)
        assert m["worst_group"] == "a"
        assert m["best_group_accuracy"] == pytest.approx(1.0)
        assert m["best_group"] == "c"
        assert m["avg_accuracy"] == pytest.approx(8 / 12)
        assert m["accuracy_gap"] == pytest.approx(8 / 12 - 0.25)
        assert m["n_groups"] == 3
        assert m["total_images"] == 12
        assert m["group_counts"] == {"a": 4, "b": 4, "c": 4}

    def test_single_group(self):
        m = group_metrics([1, 1, 0], [1, 1, 1], ["all"] * 3)
        assert m["n_groups"] == 1
        assert m["worst_group_accuracy"] == pytest.approx(m["avg_accuracy"])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            group_metrics([], [], [])


class TestCelebAGrouping:
    @pytest.mark.parametrize(
        ("blond", "male", "expected_label", "expected_group"),
        [
            (1, 1, 0, "blonde_male"),
            (1, -1, 0, "blonde_female"),
            (-1, 1, 1, "non-blonde_male"),
            (-1, -1, 1, "non-blonde_female"),
        ],
    )
    def test_minus_one_plus_one_encoding(
        self, blond, male, expected_label, expected_group
    ):
        """The mirror uses -1/+1; a `== 0` test would empty the negative groups."""
        assert ds.celeba_label_and_group(blond, male) == (expected_label, expected_group)

    def test_all_four_groups_are_reachable(self):
        groups = {ds.celeba_label_and_group(b, m)[1] for b in (1, -1) for m in (1, -1)}
        assert groups == {
            "blonde_male",
            "blonde_female",
            "non-blonde_male",
            "non-blonde_female",
        }

    def test_official_test_split_is_the_one_named_validation(self):
        """The mirror's split names are swapped relative to official CelebA."""
        assert ds.CELEBA_SPLIT == "validation"


class TestUrbanCarsGrouping:
    def test_group_label_and_class_index(self):
        assert ds.urbancars_group_label("obj-urban_bg-country_co_occur_obj-country") == (
            "obj=urban, bg=country, co=country",
            0,
        )
        assert ds.urbancars_group_label("obj-country_bg-urban_co_occur_obj-urban") == (
            "obj=country, bg=urban, co=urban",
            1,
        )

    def test_unrecognised_directory_raises(self):
        with pytest.raises(ValueError, match="unrecognised subgroup"):
            ds.urbancars_group_label("obj-suburban_bg-urban_co_occur_obj-urban")

    def test_finds_all_eight_subgroups(self, tmp_path):
        for obj in ("urban", "country"):
            for bg in ("urban", "country"):
                for co in ("urban", "country"):
                    (tmp_path / f"obj-{obj}_bg-{bg}_co_occur_obj-{co}").mkdir()
        found = ds.urbancars_subgroup_dirs(str(tmp_path))
        assert len(found) == 8
        labels = {ds.urbancars_group_label(d)[0] for d in found}
        assert len(labels) == 8

    def test_partial_layout_is_reported_not_padded(self, tmp_path):
        """A missing subgroup shrinks the group set; it is never invented."""
        (tmp_path / "obj-urban_bg-urban_co_occur_obj-urban").mkdir()
        assert len(ds.urbancars_subgroup_dirs(str(tmp_path))) == 1

    def test_missing_tree_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no UrbanCars subgroup"):
            list(ds.load_urbancars(str(tmp_path)))


class TestUrbanCarsFileSelection:
    @staticmethod
    def _tree(root, n_per_group=2):
        for obj in ("urban", "country"):
            for bg in ("urban", "country"):
                for co in ("urban", "country"):
                    d = root / f"obj-{obj}_bg-{bg}_co_occur_obj-{co}"
                    d.mkdir()
                    for i in range(n_per_group):
                        for suffix in (".jpg", "_mask.png", "_co_occur_obj_mask.png"):
                            (d / f"{i:03d}{suffix}").touch()
        return root

    def test_mask_pngs_are_not_scored(self, tmp_path):
        d = tmp_path / "obj-urban_bg-urban_co_occur_obj-urban"
        d.mkdir()
        for name in ("000.jpg", "000_mask.png", "000_co_occur_obj_mask.png"):
            (d / name).touch()
        batches = list(ds.load_urbancars(str(tmp_path)))
        paths = [p for imgs, _, _ in batches for p in imgs]
        assert paths == [str(d / "000.jpg")]

    def test_sample_count_matches_jpg_count_not_file_count(self, tmp_path):
        self._tree(tmp_path, n_per_group=5)
        batches = list(ds.load_urbancars(str(tmp_path), batch_size=64))
        paths = [p for imgs, _, _ in batches for p in imgs]
        assert len(paths) == 8 * 5
        assert all(p.endswith(".jpg") for p in paths)

    def test_every_group_is_represented(self, tmp_path):
        self._tree(tmp_path, n_per_group=3)
        batches = list(ds.load_urbancars(str(tmp_path), batch_size=64))
        groups = {g for _, _, gs in batches for g in gs}
        assert len(groups) == 8


class TestReferenceParity:
    def test_urbancars_prompts_are_frozen(self):
        assert prompts.URBANCARS_PROMPTS == {
            "urban": ["a photograph of a compact, sports, sedan car"],
            "country": ["a photograph of a truck, jeep, pickup car"],
        }

    def test_celeba_prompts_are_frozen(self):
        assert prompts.CELEBA_PROMPTS == {
            "blonde": [
                "a photo of a person with blonde hair",
                "a photo of a person with light blonde hair",
                "a photo of a person with golden hair",
                "a photo of a person with platinum blonde hair",
            ],
            "non-blonde": [
                "a photo of a person with dark hair",
                "a photo of a person with black hair",
                "a photo of a person with brown hair",
                "a photo of a brunette person",
                "a photo of a person with red hair",
                "a photo of a person with grey hair",
                "a photo of a bald person",
                "a photo of a person with auburn hair",
            ],
        }

    def test_class_name_order_matches_prompt_key_order(self):
        assert ds.CELEBA_CLASS_NAMES == tuple(prompts.CELEBA_PROMPTS)
        assert ds.URBANCARS_CLASS_NAMES == tuple(prompts.URBANCARS_PROMPTS)

    def test_urbancars_class_index_matches_reference(self):
        assert ds.URBANCARS_CLASS_NAMES.index("urban") == 0
        assert ds.URBANCARS_CLASS_NAMES.index("country") == 1

    def test_class_score_is_max_over_prompts_not_mean(self):
        torch = pytest.importorskip("torch")
        from oellm.contrib.spurious_robustness.zeroshot import class_scores, predict

        image_features = torch.tensor([[1.0, 0.0]])
        prompt_features = [
            torch.tensor([[0.9, 0.0], [0.0, 0.0]]),
            torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
        ]
        scores = class_scores(image_features, prompt_features)
        assert scores[0][0] == pytest.approx(0.9)
        assert scores[0][1] == pytest.approx(0.5)
        assert predict(scores).tolist() == [0]


class TestTasksAndAdapter:
    def test_task_properties(self):
        from oellm.contrib.spurious_robustness.task import (
            SpuriousCelebATask,
            SpuriousImageNetTask,
        )
        from oellm.contrib.spurious_urbancars.task import SpuriousUrbanCarsTask

        imagenet, celeba, urbancars = (
            SpuriousImageNetTask(),
            SpuriousCelebATask(),
            SpuriousUrbanCarsTask(),
        )
        assert imagenet.primary_metric == "top1_accuracy"
        assert celeba.primary_metric == "worst_group_accuracy"
        assert urbancars.primary_metric == "worst_group_accuracy"
        for task in (imagenet, celeba, urbancars):
            assert task.n_shots == [0]
            assert task.description

        assert imagenet.suite == "spurious_robustness"
        assert celeba.suite == "spurious_robustness"
        # UrbanCars is its own suite so its data dir can be a required env var.
        assert urbancars.suite == "spurious_urbancars"

        # Staged from the Hub; UrbanCars has nothing to stage.
        assert imagenet.dataset_specs[0].repo_id == "ILSVRC/imagenet-1k"
        assert celeba.dataset_specs[0].repo_id == "tpremoli/CelebA-attrs"
        assert urbancars.dataset_specs == []

    def test_adapter_resolves_open_clip_spec(self, tmp_path):
        assert (
            OpenClipAdapter("laion/CLIP-ViT-B-32").to_open_clip_spec()
            == "hf-hub:laion/CLIP-ViT-B-32"
        )
        # Already-prefixed and local-directory specs pass through untouched.
        assert OpenClipAdapter("hf-hub:laion/X").to_open_clip_spec() == "hf-hub:laion/X"
        assert OpenClipAdapter(str(tmp_path)).to_open_clip_spec() == str(tmp_path)

    def test_hub_suite_requires_no_cluster_env_vars(self):
        """Listing UrbanCars' path here would fail CelebA and ImageNet rows."""
        from oellm.contrib.spurious_robustness import suite

        assert suite.CLUSTER_ENV_VARS == []

    def test_urbancars_suite_declares_its_data_dir(self):
        """Declared so the login-node pre-flight catches it before submission."""
        from oellm.contrib.spurious_urbancars import suite

        assert suite.CLUSTER_ENV_VARS == ["URBANCARS_DATA_DIR"]

    def test_urbancars_without_data_dir_raises_with_guidance(self, tmp_path):
        from oellm.contrib.spurious_urbancars import suite

        with pytest.raises(RuntimeError, match="URBANCARS_DATA_DIR"):
            suite.run(
                model_path="laion/CLIP-ViT-B-32",
                task="spurious_urbancars",
                n_shot=0,
                output_path=tmp_path / "out.json",
                model_flags=None,
                env={"URBANCARS_DATA_DIR": ""},
            )

    def test_suites_do_not_claim_each_others_results(self):
        """A suite that recognises a file owns its format."""
        from oellm.contrib.spurious_robustness import suite as hub_suite
        from oellm.contrib.spurious_urbancars import suite as uc_suite

        uc = {
            "model_name_or_path": "m",
            "results": {"spurious_urbancars": {"worst_group_accuracy": 0.1}},
        }
        celeba = {
            "model_name_or_path": "m",
            "results": {"spurious_celeba": {"worst_group_accuracy": 0.2}},
        }
        assert hub_suite.parse_results(uc) is None
        assert uc_suite.parse_results(celeba) is None
        assert uc_suite.parse_results(uc)[1] == "spurious_urbancars"
        assert hub_suite.parse_results(celeba)[1] == "spurious_celeba"

    def test_login_node_preflight_checks_only_urbancars(self, monkeypatch):
        """The whole point of the split: no false failures for the Hub tasks."""
        import sys
        from pathlib import Path

        from oellm.envcheck import collect_problems

        monkeypatch.delenv("URBANCARS_DATA_DIR", raising=False)
        venv = str(Path(sys.prefix))

        problems = collect_problems({"spurious_urbancars"}, venv_path=venv)
        assert any("URBANCARS_DATA_DIR" in p for p in problems)
        assert collect_problems({"spurious_robustness"}, venv_path=venv) == []

    def test_unknown_task_raises(self, tmp_path):
        from oellm.contrib.spurious_robustness import suite

        with pytest.raises(ValueError, match="Unknown task"):
            suite.run(
                model_path="laion/CLIP-ViT-B-32",
                task="spurious_nonexistent",
                n_shot=0,
                output_path=tmp_path / "out.json",
                model_flags=None,
                env={},
            )


class TestSchedule:
    """Render-level checks: the SLURM script and jobs.csv the scheduler emits."""

    def _schedule(self, tmp_path, group):
        import os
        import sys
        from pathlib import Path
        from unittest.mock import patch

        from oellm.main import schedule_evals

        with (
            patch("oellm.scheduler._load_cluster_env"),
            patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
            patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        ):
            schedule_evals(
                models="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                task_groups=group,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
            )

    def test_dry_run_routes_to_contrib_dispatch(self, tmp_path):
        self._schedule(tmp_path, "spurious-celeba")
        sbatch = list(tmp_path.glob("**/submit_evals.sbatch"))
        assert len(sbatch) == 1
        assert "oellm.contrib.dispatch" in sbatch[0].read_text()

    def test_jobs_csv_carries_the_suite_and_task(self, tmp_path):
        import pandas as pd

        self._schedule(tmp_path, "spurious-robustness,spurious-urbancars")
        csvs = list(tmp_path.glob("**/jobs.csv"))
        assert len(csvs) == 1
        df = pd.read_csv(csvs[0])
        # The frozen jobs.csv schema.
        assert list(df.columns) == ["model_path", "task_path", "n_shot", "eval_suite"]
        assert set(df["task_path"]) == {
            "spurious_imagenet",
            "spurious_celeba",
            "spurious_urbancars",
        }
        # Both groups schedule together; each row carries its own suite so the
        # dispatcher routes UrbanCars to the suite that requires its data dir.
        by_task = dict(zip(df["task_path"], df["eval_suite"], strict=True))
        assert by_task["spurious_imagenet"] == "spurious_robustness"
        assert by_task["spurious_celeba"] == "spurious_robustness"
        assert by_task["spurious_urbancars"] == "spurious_urbancars"
        assert set(df["n_shot"]) == {0}


class TestImageNetFolderLoader:
    """The local ImageFolder path, whose class indices must match OpenCLIP's order."""

    def _tree(self, tmp_path, n_synsets, images_per_synset=1):
        # Synset ids are not contiguous in reality; what matters is that ascending
        # synset order is the canonical class order.
        for i in range(n_synsets):
            d = tmp_path / f"n{i:08d}"
            d.mkdir()
            for j in range(images_per_synset):
                (d / f"img_{j}.JPEG").write_bytes(b"")
        return tmp_path

    def test_label_index_follows_sorted_synset_order(self, tmp_path):
        root = self._tree(tmp_path, 1000)
        batches = list(ds.load_imagenet(str(root), batch_size=256))
        paths = [p for b in batches for p in b[0]]
        labels = [lbl for b in batches for lbl in b[1]]
        groups = [g for b in batches for g in b[2]]

        assert len(labels) == 1000
        # Ascending synset directory order == ascending class index.
        assert labels == sorted(labels)
        assert labels[0] == 0 and labels[-1] == 999
        assert "n00000000" in paths[0] and "n00000999" in paths[-1]
        # ImageNet has no spurious attribute: exactly one group.
        assert set(groups) == {"all"}

    def test_limit_is_honoured(self, tmp_path):
        root = self._tree(tmp_path, 1000)
        labels = [lbl for b in ds.load_imagenet(str(root), limit=10) for lbl in b[1]]
        assert len(labels) == 10

    def test_wrong_class_count_raises(self, tmp_path):
        """A partial tree would silently shift every class index."""
        root = self._tree(tmp_path, 3)
        with pytest.raises(ValueError, match="expected 1000 synset directories"):
            list(ds.load_imagenet(str(root)))
