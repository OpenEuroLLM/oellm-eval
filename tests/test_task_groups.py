from oellm.task_groups import (
    TaskGroup,
    _build_task_suite_map,
    _expand_lang_templates,
    _expand_task_groups,
    _load_task_groups_data,
    primary_metric_map,
)


def test_group_subset_default_and_task_override():
    group = TaskGroup.from_dict(
        "example",
        {
            "description": "Subset inheritance",
            "suite": "lm-eval-harness",
            "n_shots": [0],
            "dataset": "example/dataset",
            "subset": "english",
            "tasks": [
                {"task": "inherited"},
                {"task": "overridden", "subset": "french"},
            ],
        },
    )
    assert [task.subset for task in group.tasks] == ["english", "french"]


class TestExpandLangTemplates:
    def test_no_valid_langs_unchanged(self):
        data = {
            "task_groups": {
                "my-group": {
                    "suite": "lm-eval-harness",
                    "tasks": [{"task": "arc_challenge", "n_shots": [0]}],
                }
            }
        }
        result = _expand_lang_templates(data)
        assert result == data

    def test_task_template_expanded(self):
        data = {
            "task_groups": {
                "my-group": {
                    "suite": "lm-eval-harness",
                    "valid_langs": ["en", "de"],
                    "tasks": [{"task": "foo_{lang}", "subset": "{lang}"}],
                }
            }
        }
        result = _expand_lang_templates(data)
        tasks = result["task_groups"]["my-group"]["tasks"]
        assert tasks == [
            {"task": "foo_en", "subset": "en"},
            {"task": "foo_de", "subset": "de"},
        ]

    def test_valid_langs_key_removed(self):
        data = {
            "task_groups": {
                "my-group": {
                    "suite": "lm-eval-harness",
                    "valid_langs": ["en"],
                    "tasks": [{"task": "foo_{lang}"}],
                }
            }
        }
        result = _expand_lang_templates(data)
        assert "valid_langs" not in result["task_groups"]["my-group"]

    def test_non_template_tasks_preserved(self):
        data = {
            "task_groups": {
                "my-group": {
                    "suite": "lm-eval-harness",
                    "valid_langs": ["en", "de"],
                    "tasks": [
                        {"task": "foo_{lang}", "subset": "{lang}"},
                        {"task": "bar_static"},
                    ],
                }
            }
        }
        result = _expand_lang_templates(data)
        tasks = result["task_groups"]["my-group"]["tasks"]
        assert len(tasks) == 3
        assert tasks[0] == {"task": "foo_en", "subset": "en"}
        assert tasks[1] == {"task": "foo_de", "subset": "de"}
        assert tasks[2] == {"task": "bar_static"}

    def test_does_not_mutate_input(self):
        data = {
            "task_groups": {
                "my-group": {
                    "suite": "lm-eval-harness",
                    "valid_langs": ["en"],
                    "tasks": [{"task": "foo_{lang}"}],
                }
            }
        }
        _expand_lang_templates(data)
        assert "valid_langs" in data["task_groups"]["my-group"]
        assert data["task_groups"]["my-group"]["tasks"] == [{"task": "foo_{lang}"}]

    def test_task_only_template_no_subset(self):
        """Template in task name only, no subset field."""
        data = {
            "task_groups": {
                "flores": {
                    "suite": "lighteval",
                    "valid_langs": ["bul_Cyrl", "ces_Latn"],
                    "tasks": [{"task": "flores200:{lang}-eng_Latn"}],
                }
            }
        }
        result = _expand_lang_templates(data)
        tasks = result["task_groups"]["flores"]["tasks"]
        assert tasks == [
            {"task": "flores200:bul_Cyrl-eng_Latn"},
            {"task": "flores200:ces_Latn-eng_Latn"},
        ]


class TestLoadTaskGroupsData:
    def test_returns_dict_with_task_groups(self):
        data = _load_task_groups_data()
        assert "task_groups" in data
        assert isinstance(data["task_groups"], dict)

    def test_no_valid_langs_in_expanded_data(self):
        """After loading, no task group should have valid_langs remaining."""
        data = _load_task_groups_data()
        for name, group in data["task_groups"].items():
            assert "valid_langs" not in group, f"valid_langs still present in {name}"

    def test_no_lang_placeholder_in_task_names(self):
        """After loading, no task entry should contain the {lang} placeholder."""
        data = _load_task_groups_data()
        for group_name, group in data["task_groups"].items():
            for task in group.get("tasks", []):
                assert "{lang}" not in task.get("task", ""), (
                    f"Unexpanded {{lang}} in task '{task['task']}' of group '{group_name}'"
                )


class TestBackwardCompatibility:
    def test_open_sci_task_count(self):
        """Groups without valid_langs (e.g. open-sci-0.01) are unaffected."""
        results = _expand_task_groups(["open-sci-0.01"])
        # 12 tasks, hellaswag has n_shots=[10] => 12 results
        assert len(results) == 12

    def test_open_sci_no_lang_placeholders(self):
        results = _expand_task_groups(["open-sci-0.01"])
        for r in results:
            assert "{lang}" not in r.task

    def test_open_sci_task_names_unchanged(self):
        results = _expand_task_groups(["open-sci-0.01"])
        task_names = {r.task for r in results}
        assert "arc_challenge" in task_names
        assert "hellaswag" in task_names
        assert "mmlu" in task_names


class TestExpandTaskGroupsWithTemplates:
    def test_sib200_expands_to_36_tasks(self):
        results = _expand_task_groups(["sib200-eu"])
        assert len(results) == 36

    def test_belebele_cf_expands_to_26_tasks(self):
        results = _expand_task_groups(["belebele-eu-cf"])
        assert len(results) == 26

    def test_arc_challenge_mt_expands_to_22_tasks(self):
        # 21 templated + 1 static (arc_challenge_mt_is)
        results = _expand_task_groups(["arc-challenge-mt-eu"])
        assert len(results) == 22
        task_names = {r.task for r in results}
        assert "arc_challenge_mt_is" in task_names
        assert "arc_challenge_mt_bg" in task_names

    def test_flores_eu_to_eng_expands_to_35_tasks(self):
        results = _expand_task_groups(["flores-200-eu-to-eng"])
        assert len(results) == 35
        assert {result.n_shot for result in results} == {4}

    def test_flores_eng_to_eu_uses_four_shots(self):
        results = _expand_task_groups(["flores-200-eng-to-eu"])

        assert len(results) == 35
        assert {result.n_shot for result in results} == {4}

    def test_global_mmlu_expands_to_16_tasks(self):
        """16, not 18: Russian and Hebrew are not OpenEuroLLM target languages."""
        results = _expand_task_groups(["global-mmlu-eu"])
        assert len(results) == 16
        task_names = {r.task for r in results}
        assert "global_mmlu_full_ru" not in task_names
        assert "global_mmlu_full_he" not in task_names

    def test_global_piqa_completions_expands_to_32_tasks(self):
        results = _expand_task_groups(["global-piqa-eu-completions"])
        assert len(results) == 32


class TestFlagEvals:
    def test_halves_split_by_backend(self):
        vllm = _expand_task_groups(["flag-evals-vllm"])
        lighteval = _expand_task_groups(["flag-evals-lighteval"])
        assert (len(vllm), len(lighteval)) == (365, 73)
        assert {r.suite for r in vllm} == {"lm-eval-harness", "evalchemy"}
        assert {r.suite for r in lighteval} == {"lighteval"}
        assert len(_expand_task_groups(["flag-evals"])) == 438

    def test_sib200_reports_acc(self):
        metrics = primary_metric_map()
        assert {metrics[r.task] for r in _expand_task_groups(["sib200-eu"])} == {"acc"}


def _evals(group):
    return {(r.task, r.n_shot) for r in _expand_task_groups([group])}


class TestFlagTiers:
    def test_tier_sizes(self):
        sizes = {
            f"{tier}-{half}": len(_evals(f"flag-{tier}-{half}"))
            for tier, halves in [("t1", "vllm lighteval"), ("t2", "vllm lighteval cot"),
                                 ("t3", "vllm lighteval cot"), ("t4", "vllm lighteval")]
            for half in halves.split()
        }
        assert sizes == {"t1-vllm": 132, "t1-lighteval": 12, "t2-vllm": 266, "t2-lighteval": 73,
                         "t2-cot": 14, "t3-vllm": 330, "t3-lighteval": 73, "t3-cot": 32,
                         "t4-vllm": 154, "t4-lighteval": 14}

    def test_each_tier_contains_the_one_below(self):
        for half in ("vllm", "lighteval"):
            assert _evals(f"flag-t1-{half}") <= _evals(f"flag-t2-{half}") <= _evals(f"flag-t3-{half}")
        assert _evals("flag-t2-cot") <= _evals("flag-t3-cot")
        assert _evals("flag-t1-vllm") <= _evals("flag-t4-vllm")
        assert _evals("flag-t1-lighteval") <= _evals("flag-t4-lighteval")

    def test_held_out_groups_only_in_tier_3(self):
        held_out = set().union(*(_evals(g) for g in [
            "include", "gsm8k-cot", "mmlu-prox-eu", "global-piqa-eu-prompted",
            "flag-cot-math", "flag-cot-polymath-high", "flag-cot-polymath-top"]))
        decide = set().union(*(_evals(f"flag-t2-{h}") for h in ("vllm", "lighteval", "cot")))
        report = set().union(*(_evals(f"flag-t3-{h}") for h in ("vllm", "lighteval", "cot")))
        assert held_out <= report
        assert held_out.isdisjoint(decide)

    def test_global_mgsm_only_post_train(self):
        base = set().union(*(_evals(f"flag-t3-{h}") for h in ("vllm", "lighteval", "cot")))
        assert _evals("global-mgsm-eu").isdisjoint(base)
        assert _evals("global-mgsm-eu") <= _evals("flag-t4-vllm")

    def test_no_livecodebench_in_tiers(self):
        for group in ("flag-t2-cot", "flag-t3-cot"):
            assert "livecodebench_cont" not in {task for task, _ in _evals(group)}

    def test_shots(self):
        assert _evals("gsm8k-cot") == {("gsm8k_cot", 8)}
        assert {n for _, n in _evals("global-mgsm-eu")} == {0}


def test_judgearena_does_not_register_standalone_ja_variants():
    assert {"alpaca-eval-ja", "arena-hard-v0.1-ja", "arena-hard-v2.0-ja"}.isdisjoint(
        _build_task_suite_map()
    )


def test_judgearena_elo_categories():
    en = {result.task for result in _expand_task_groups(["elo-en"])}
    eu = {result.task for result in _expand_task_groups(["elo-eu"])}
    multilingual = {result.task for result in _expand_task_groups(["elo-multilingual"])}

    assert en == {
        "elo-comparia-en",
        "elo-lmarena-en",
        "elo-lmarena-100k-en",
        "elo-lmarena-140k-en",
    }
    assert en <= eu <= multilingual
    assert len(eu) == 32
    assert len(multilingual) == 53
    assert {result.task for result in _expand_task_groups(["elo-eu[deu_Latn]"])} == {
        "elo-lmarena-de",
        "elo-lmarena-100k-de",
        "elo-lmarena-140k-de",
    }
    assert {
        (result.suite, result.n_shot) for result in _expand_task_groups(["elo-eu"])
    } == {("judgearena", 0)}


def test_judgearena_additional_routes_only_three_tasks():
    assert {
        (result.task, result.suite, result.n_shot)
        for result in _expand_task_groups(["judgearena-additional"])
    } == {
        ("mt-bench-101", "judgearena", 0),
        ("m-arena-hard-v0.1-EU", "judgearena", 0),
        ("m-arena-hard-v2.0-EU", "judgearena", 0),
    }
