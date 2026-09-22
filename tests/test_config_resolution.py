"""The SLURM scripts and run_agent must agree on what a --config means.

They did not. `run_agent` prepended "configs/" unconditionally, while
`load_serving_config` tried the path as given first. So the submit script,
which builds its list by globbing `configs/model_size/*.json`, produced a path
that serving_params accepted and run_agent turned into
`configs/configs/model_size/...`. The mismatch only surfaced on the cluster,
after a model had been loaded, disguised as "the server shut down".
"""

import subprocess
import sys

import pytest

from belief_update_sim.config import CONFIGS_DIR, PROJECT_ROOT, resolve_config_path
from belief_update_sim.serving import load_serving_config

# Exactly what `for c in configs/model_size/*.json` yields from the repo root.
GLOB_FORM = "configs/model_size/Qwen3.5-9B.json"
# Exactly what `--config` has always meant.
NAME_FORM = "model_size/Qwen3.5-9B.json"


def test_name_under_configs_resolves_into_configs():
    assert resolve_config_path("gpt-5.2.json") == CONFIGS_DIR / "gpt-5.2.json"
    assert resolve_config_path(NAME_FORM) == CONFIGS_DIR / "model_size/Qwen3.5-9B.json"


def test_an_existing_path_is_left_alone(tmp_path, monkeypatch):
    """The bug: this form used to get 'configs/' prepended a second time."""
    monkeypatch.chdir(PROJECT_ROOT)
    resolved = resolve_config_path(GLOB_FORM)
    assert resolved.exists()
    assert "configs/configs" not in resolved.as_posix()
    assert "configs\\configs" not in str(resolved)


def test_absolute_paths_are_left_alone():
    absolute = CONFIGS_DIR / "model_size" / "Qwen3.5-9B.json"
    assert resolve_config_path(absolute) == absolute


def test_both_forms_name_the_same_file(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    assert (resolve_config_path(GLOB_FORM).resolve()
            == resolve_config_path(NAME_FORM).resolve())


def test_serving_accepts_both_forms(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    assert load_serving_config(GLOB_FORM).model == load_serving_config(NAME_FORM).model


@pytest.mark.parametrize("form", [GLOB_FORM, NAME_FORM])
def test_run_agent_accepts_both_forms(form):
    """End to end: run_agent must get past config loading for either spelling.

    It cannot reach the model (no server), so the check is that it fails
    *later* than config loading -- never with FileNotFoundError on the config.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.run_agent",
         "--config", form, "--pilot", "--limit", "1", "--single_topic", "UBI"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=180,
    )
    combined = result.stdout + result.stderr
    assert "configs/configs" not in combined.replace("\\", "/")
    assert "No such file or directory" not in combined
    assert "Config not found" not in combined
    # It got far enough to read the config and name its output path.
    assert "Load config from" in combined


def test_a_genuinely_missing_config_fails_before_anything_else():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.run_agent",
         "--config", "model_size/does-not-exist.json", "--pilot"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
    )
    assert result.returncode != 0
    assert "Config not found" in result.stdout + result.stderr


def test_every_sweep_config_resolves_from_the_glob_form(monkeypatch):
    """Guards the whole batch, not just the one model I happened to test."""
    monkeypatch.chdir(PROJECT_ROOT)
    for path in sorted(CONFIGS_DIR.glob("model_size/*.json")):
        glob_form = path.relative_to(PROJECT_ROOT).as_posix()
        assert resolve_config_path(glob_form).exists(), glob_form
        assert load_serving_config(glob_form).model == path.stem
