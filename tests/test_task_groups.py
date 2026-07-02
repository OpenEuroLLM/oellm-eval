from oellm.task_groups import (
    _expand_lang_templates,
    _expand_task_groups,
    _load_task_groups_data,
)


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

    def test_global_mmlu_expands_to_18_tasks(self):
        results = _expand_task_groups(["global-mmlu-eu"])
        assert len(results) == 18

    def test_global_piqa_completions_expands_to_32_tasks(self):
        results = _expand_task_groups(["global-piqa-eu-completions"])
        assert len(results) == 32
