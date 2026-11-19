from unittest.mock import patch

import pytest

from oellm.bbq import _target_locations, bbq_metrics, prepare_mapping


def sample(pred, context="ambig", target="C", source_id=0):
    return {
        "id": f"Age_{source_id:05}",
        "target": target,
        "scores": {"choice": {"answer": pred}},
        "metadata": {
            "category": "Age",
            "context_condition": context,
            "source_id": source_id,
            "question_index": "1",
            "answer_info": {
                "ans0": ["older", "old"],
                "ans1": ["younger", "nonOld"],
                "ans2": ["unknown", "unknown"],
            },
        },
    }


def metric(rows, name, context="ambig"):
    with patch("oellm.bbq._target_locations", return_value={("Age", "1", "0"): 0}):
        return bbq_metrics(rows)[f"bbq/Age/{context}/{name}"]


def test_bias_is_accuracy_scaled_only_for_ambiguous_context():
    answers = ["A", "A", "B", "C"]
    assert metric([sample(a) for a in answers], "bias_score") == pytest.approx(0.25)
    assert metric(
        [sample(a, "disambig", "A") for a in answers], "bias_score", "disambig"
    ) == pytest.approx(1 / 3)


def test_unknown_and_invalid_are_distinct():
    assert metric([sample("C")], "bias_score") == 0
    assert metric([sample("C", "disambig", "A")], "bias_score", "disambig") is None
    assert metric([sample("invalid")], "bias_score") is None
    assert metric([sample("invalid")], "n_invalid") == 1


def test_unmapped_samples_report_coverage_and_do_not_change_bias():
    rows = [sample("A"), sample("B", source_id=999)]
    assert metric(rows, "mapping_coverage") == 0.5
    assert metric(rows, "n_unmapped") == 1
    assert metric(rows, "bias_score") == 1


def test_missing_metadata():
    with pytest.raises(ValueError, match="saved sample"):
        bbq_metrics([])
    with patch("oellm.bbq._target_locations", return_value={}):
        with pytest.raises(ValueError, match="category/context"):
            bbq_metrics([{}])


def test_prepare_and_load_verified_mapping(tmp_path, monkeypatch):
    import hashlib
    import io

    import oellm.bbq as bbq

    data = b"category,question_index,example_id,target_loc\nAge,1,0,0\nAge,1,1,NA\n"
    path = tmp_path / "metadata.csv"
    monkeypatch.setenv("OELLM_BBQ_METADATA", str(path))
    monkeypatch.setattr(bbq, "SOURCE_SHA256", hashlib.sha256(data).hexdigest())
    _target_locations.cache_clear()
    with pytest.raises(FileNotFoundError, match="python -m oellm.bbq"):
        _target_locations()
    with patch("oellm.bbq.urlopen", return_value=io.BytesIO(data)) as download:
        assert prepare_mapping() == path
        assert prepare_mapping() == path
        download.assert_called_once()
    assert _target_locations() == {("Age", "1", "0"): 0, ("Age", "1", "1"): None}
    path.write_bytes(b"corrupted")
    _target_locations.cache_clear()
    with pytest.raises(ValueError, match="checksum"):
        _target_locations()
    path.unlink()
    with patch("oellm.bbq.urlopen", return_value=io.BytesIO(b"corrupted")):
        with pytest.raises(ValueError, match="checksum"):
            prepare_mapping()
    assert not path.exists()
    _target_locations.cache_clear()
