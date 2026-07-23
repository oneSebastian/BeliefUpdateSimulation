"""Structural tests for the HF export.

The registry/coverage tests read the real results tree, since their whole point
is to catch drift between the registry and what is on disk. The record-shaping
tests use small synthetic frames so they run without touching the 600MB export.
"""

import os

import pandas as pd
import pytest

from scripts.data import export_hf_dataset as ex

EXPECTED_SPLITS = {"main", "simulated_initial", "ablations"}


def test_registry_accounts_for_every_results_file():
    """Fails when a new results file is added without deciding its fate."""
    ex.check_registry_covers_disk()


def test_registry_and_exclusions_are_disjoint():
    registered = {os.path.relpath(e["path"], ex.ROOT).replace("\\", "/") for e in ex.REGISTRY}
    assert not (registered & set(ex.EXCLUDED))


def test_registry_paths_are_unique():
    paths = [e["path"] for e in ex.REGISTRY]
    assert len(paths) == len(set(paths))


def test_only_expected_splits_are_used():
    assert {e["split"] for e in ex.REGISTRY} == EXPECTED_SPLITS


def test_split_composition():
    counts = {}
    for e in ex.REGISTRY:
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    assert counts == {"main": 6, "simulated_initial": 6, "ablations": 21}


def test_ablation_entries_are_all_typed():
    for e in ex.REGISTRY:
        if e["split"] == "ablations":
            assert e.get("ablation_type"), f"{e['path']} has no ablation_type"
            assert e.get("condition"), f"{e['path']} has no condition"


def test_non_ablation_entries_carry_no_condition():
    for e in ex.REGISTRY:
        if e["split"] != "ablations":
            assert e.get("ablation_type") is None
            assert e.get("condition") is None


def test_canonical_model_names_are_used():
    """Guards against a raw in-file name (e.g. academic-ai-gpt-5-mini) leaking
    into the registry."""
    allowed = {ex.GPT52, ex.GPT5MINI, ex.GEMINI, ex.CLAUDE, ex.QWEN, ex.LLAMA} | {
        "OLMo-3-32B-Think", "OLMo-3-32B-Think-SFT", "OLMo-3-32B-Think-DPO",
        "OLMo-3.1-32B-Instruct", "OLMo-3.1-32B-Instruct-SFT", "OLMo-3.1-32B-Instruct-DPO",
    }
    assert {e["model"] for e in ex.REGISTRY} <= allowed


def test_same_six_models_in_main_and_simulated_initial():
    main = {e["model"] for e in ex.REGISTRY if e["split"] == "main"}
    sim = {e["model"] for e in ex.REGISTRY if e["split"] == "simulated_initial"}
    assert main == sim


def test_temperature_entries_carry_a_temperature():
    for e in ex.REGISTRY:
        if e.get("ablation_type") == "temperature":
            assert e.get("temperature") in (0.0, 2.0)


def test_probe_files_are_excluded_not_exported():
    probes = [p for p in ex.EXCLUDED if "probe-initial-belief" in p]
    assert len(probes) == 6


# --- record shaping --------------------------------------------------------

POS = "Everyone should receive Universal Basic Income."
NEG = "There should be no Universal Basic Income."


def _write_xlsx(tmp_path, rows):
    path = tmp_path / "fixture.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return str(path)


def _row(formulation, **kw):
    row = {
        "persona_id": "p1", "topic": "UBI", "model": "raw-name",
        "statement_formulation": formulation, "init_belief": 1, "familiarity": 0,
        "package": "package1", "message_order_index": 0,
        "message_order_perm": "[1, 2, 3]", "shown_messages": '[{"shown_comment_id": 1}]',
        "demographic": '{"age": 40}', "prompt_sent": "prompt",
        "llm_new_belief": 2, "rank_1": 1, "rank_2": 2, "rank_3": 3,
        "llm_general_public_stance": -1, "llm_reasoning": "because",
        "raw_response": "{}",
    }
    row.update(kw)
    return row


def test_positive_framing_leaves_raw_and_normalized_equal(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT52)
    records, _ = ex.load_llm_file(entry)
    r = records[0]
    assert r["statement_polarity"] == 1
    assert r["raw_init_belief"] == r["normalized_init_belief"] == 1
    assert r["raw_new_belief"] == r["normalized_new_belief"] == 2


def test_negative_framing_flips_all_three_belief_fields(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(NEG)]), split="main", model=ex.GPT52)
    r = ex.load_llm_file(entry)[0][0]
    assert r["statement_polarity"] == -1
    assert (r["raw_init_belief"], r["normalized_init_belief"]) == (1, -1)
    assert (r["raw_new_belief"], r["normalized_new_belief"]) == (2, -2)
    assert (r["raw_general_public_stance"], r["normalized_general_public_stance"]) == (-1, 1)


def test_ranks_are_never_flipped(tmp_path):
    """Ranks index which message was shown, not a polarity."""
    entry = dict(path=_write_xlsx(tmp_path, [_row(NEG)]), split="main", model=ex.GPT52)
    r = ex.load_llm_file(entry)[0][0]
    assert (r["rank_1"], r["rank_2"], r["rank_3"]) == (1, 2, 3)


def test_embedded_json_is_parsed_into_objects(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT52)
    r = ex.load_llm_file(entry)[0][0]
    assert r["message_order_perm"] == [1, 2, 3]
    assert isinstance(r["shown_messages"], list)


def test_demographic_always_has_the_canonical_key_set(tmp_path):
    """Sources carry 9, 11 or 19 demographic keys; HF needs one struct shape."""
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT52)
    demographic = ex.load_llm_file(entry)[0][0]["demographic"]
    assert set(demographic) == set(ex.DEMO_KEYS) | {"big5", "big5_items"}
    assert set(demographic["big5"]) == set(ex.BIG5_TRAITS)
    assert len(demographic["big5_items"]) == 10


def test_shown_messages_have_one_struct_shape_for_both_sources():
    """Model files store dicts, the human CSV stores bare ids. Emitting both
    as-is gives list<struct> vs list<int64> in one split, which HF cannot cast."""
    fields = {"shown_comment_id", "source_message_id", "package", "text"}
    texts = {("UBI", "package1", 3): "the message text"}

    from_model = ex.canonical_messages(
        [{"shown_comment_id": 1, "source_message_id": 3, "package": "package1", "text": "t"}],
        "UBI", "package1", texts)
    from_human = ex.canonical_messages([3], "UBI", "package1", texts)

    assert set(from_model[0]) == set(from_human[0]) == fields
    assert from_human[0]["source_message_id"] == 3
    assert from_human[0]["shown_comment_id"] == 1


def test_human_messages_recover_text_from_the_message_index():
    texts = {("UBI", "package1", 2): "recovered wording"}
    out = ex.canonical_messages([2], "UBI", "package1", texts)
    assert out[0]["text"] == "recovered wording"


def test_unknown_message_slot_yields_null_text_not_an_error():
    out = ex.canonical_messages([9], "UBI", "package1", {})
    assert out[0]["text"] is None


# --- schema ----------------------------------------------------------------

def test_feature_spec_matches_emitted_llm_records(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT52)
    record = ex.load_llm_file(entry)[0][0]
    assert [name for name, _ in ex.FEATURE_SPEC] == list(record)


def test_feature_spec_covers_nested_demographic_keys(tmp_path):
    spec = dict(ex.FEATURE_SPEC)["demographic"]
    assert spec[0] == "struct"
    names = [n for n, _ in spec[1]]
    assert set(names) == set(ex.DEMO_KEYS) | {"big5", "big5_items"}


def test_big5_scores_are_float_not_int():
    """They are two-item means, so half-points occur; declaring int64 makes
    pyarrow reject 3.5."""
    demographic = dict(ex.FEATURE_SPEC)["demographic"]
    big5 = dict(demographic[1])["big5"]
    assert all(dtype == "float64" for _, dtype in big5[1])


def test_temperature_is_declared_float_not_inferred():
    """Null until row 7039 of the ablations split, so inference guesses null."""
    assert dict(ex.FEATURE_SPEC)["temperature"] == "float64"


def test_features_build_as_a_datasets_object():
    pytest.importorskip("datasets")
    features = ex.hf_features()
    assert set(features) == {name for name, _ in ex.FEATURE_SPEC}


def test_dataset_card_declares_the_splits():
    card = ex.dataset_card({"main": [1], "simulated_initial": [1], "ablations": [1]})
    for split in ("main", "simulated_initial", "ablations"):
        assert f"split: {split}" in card
        assert f"path: {split}.jsonl" in card


def test_dataset_card_leaves_column_types_to_the_hub():
    """Types are inferred from the data rather than declared in the card."""
    card = ex.dataset_card({"main": [1], "simulated_initial": [1], "ablations": [1]})
    assert "dataset_info:" not in card
    assert "dtype:" not in card


def test_dataset_card_states_licence_and_ethics():
    card = ex.dataset_card({"main": [1], "simulated_initial": [1], "ablations": [1]})
    assert "license: cc-by-4.0" in card
    assert "2025-09" in card                 # ethics committee case number
    assert "re-identify" in card


def test_canonical_model_overrides_the_in_file_name(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT5MINI)
    assert ex.load_llm_file(entry)[0][0]["model"] == ex.GPT5MINI


def test_free_text_gpt_is_read_as_reasoning(tmp_path):
    row = _row(POS)
    del row["llm_reasoning"]
    row["free_text_gpt"] = "gpt reasoning"
    entry = dict(path=_write_xlsx(tmp_path, [row]), split="main", model=ex.GPT52)
    records, repaired = ex.load_llm_file(entry)
    assert records[0]["reasoning"] == "gpt reasoning"
    assert repaired == 0  # aliased, not recovered from raw_response


def test_simulated_split_marks_belief_origin(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="simulated_initial", model=ex.GPT52)
    assert ex.load_llm_file(entry)[0][0]["init_belief_source"] == "simulated"

    entry = dict(path=_write_xlsx(tmp_path, [_row(POS)]), split="main", model=ex.GPT52)
    assert ex.load_llm_file(entry)[0][0]["init_belief_source"] == "participant"


def test_unknown_formulation_aborts_the_export(tmp_path):
    """Better to fail than to publish a silently mis-signed row."""
    from belief_update_sim.normalization import UnknownFormulationError
    entry = dict(path=_write_xlsx(tmp_path, [_row("A brand new statement.")]),
                 split="main", model=ex.GPT52)
    with pytest.raises(UnknownFormulationError):
        ex.load_llm_file(entry)


def test_null_beliefs_survive_as_null(tmp_path):
    entry = dict(path=_write_xlsx(tmp_path, [_row(NEG, llm_new_belief=None, raw_response="x")]),
                 split="main", model=ex.GPT52)
    r = ex.load_llm_file(entry)[0][0]
    assert r["raw_new_belief"] is None
    assert r["normalized_new_belief"] is None
