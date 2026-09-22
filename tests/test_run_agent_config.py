"""Tests for the output-path guard in scripts/pipeline/run_agent.py.

Configs for completed runs rename `output_excel` to `disabled_output_excel` so a
re-run cannot overwrite published results. That used to surface as a bare
KeyError; it must now be an explicit, actionable message.
"""
import json

import pytest

from belief_update_sim.config import CONFIGS_DIR
from scripts.pipeline.run_agent import resolve_output_excel


def test_enabled_config_returns_path():
    paths = {"output_excel": "results/model_all_personas_results.xlsx"}
    assert resolve_output_excel(paths, "configs/x.json") == paths["output_excel"]


@pytest.mark.parametrize("key", ["disabled_output_excel", "disabeled_output_excel"])
def test_disabled_config_exits_with_guidance(key):
    """Both the corrected and the historical misspelling must be recognised."""
    paths = {key: "results/model_all_personas_results.xlsx"}
    with pytest.raises(SystemExit) as exc:
        resolve_output_excel(paths, "configs/x.json")

    message = str(exc.value)
    assert "disabled for configs/x.json" in message
    assert "results/model_all_personas_results.xlsx" in message
    assert "output_excel" in message          # tells the reader how to re-enable
    assert "--ablation" in message            # and that ablations still work


def test_missing_output_key_exits_and_lists_what_was_found():
    paths = {"personas_root": "data/prolific_data"}
    with pytest.raises(SystemExit) as exc:
        resolve_output_excel(paths, "configs/x.json")
    assert "no output path" in str(exc.value)
    assert "personas_root" in str(exc.value)


# rglob, not glob: configs are grouped into subdirectories (configs/model_size/)
# and a non-recursive glob would quietly stop covering them.
def all_configs():
    return sorted(CONFIGS_DIR.rglob("*.json"))


def test_configs_are_discovered_in_subdirectories():
    found = {p.relative_to(CONFIGS_DIR).as_posix() for p in all_configs()}
    assert any(name.startswith("model_size/") for name in found), found


def test_no_config_uses_the_misspelled_key():
    """The corrected spelling is what ships; the alias exists only for safety."""
    offenders = [c.name for c in all_configs()
                 if "disabeled_output_excel" in c.read_text(encoding="utf-8-sig")]
    assert offenders == []


def test_every_config_declares_exactly_one_output_key():
    for config_path in all_configs():
        entries = json.loads(config_path.read_bytes().decode("utf-8-sig"))
        for entry in entries:
            paths = entry["paths"]
            declared = [k for k in paths
                        if k in ("output_excel", "disabled_output_excel")]
            assert len(declared) == 1, f"{config_path.name}: {sorted(paths)}"


def test_every_config_resolves_or_exits_cleanly():
    """No config may raise anything other than SystemExit."""
    for config_path in all_configs():
        entries = json.loads(config_path.read_bytes().decode("utf-8-sig"))
        for entry in entries:
            try:
                resolve_output_excel(entry["paths"], config_path.name)
            except SystemExit:
                pass
