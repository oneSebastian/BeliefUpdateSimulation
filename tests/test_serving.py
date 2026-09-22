"""Tests for the `serving` block that sizes the SLURM allocation.

These parameters are only exercised on the cluster, where a mistake costs a
queue slot and a model load before it surfaces. Validating them here means a
typo fails on the laptop instead.
"""

import json
import shlex
import subprocess
import sys

import pytest

from belief_update_sim.config import CONFIGS_DIR, PROJECT_ROOT
from belief_update_sim.serving import (
    MAX_GPUS,
    PARTITION_TIME_LIMITS_HOURS,
    PROMPT_HEADROOM_TOKENS,
    ServingConfig,
    ServingConfigError,
    load_serving_config,
    parse_slurm_time,
)

MODEL_SIZE_DIR = CONFIGS_DIR / "model_size"


def make(**overrides) -> ServingConfig:
    """A valid ServingConfig, with fields overridden per test."""
    base = dict(
        config_name="test.json",
        model="Test-9B",
        hf_model_id="org/Test-9B",
        gpus=1,
        tensor_parallel_size=1,
        max_model_len=20480,
        max_tokens=16384,
        gpu_memory_utilization=0.90,
        port=8100,
        partition="verylong",
        time_limit="12:00:00",
    )
    base.update(overrides)
    return ServingConfig(**base)


def write_config(tmp_path, **serving_overrides):
    serving = dict(
        hf_model_id="org/Test-9B",
        gpus=1,
        tensor_parallel_size=1,
        max_model_len=20480,
        gpu_memory_utilization=0.9,
        partition="verylong",
        time="12:00:00",
    )
    serving.update(serving_overrides)
    entry = {"model": "Test-9B", "max_tokens": 16384, "port": 8100,
             "serving": serving, "paths": {}}
    path = tmp_path / "test.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# walltime parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,hours", [
    ("01:00:00", 1),
    ("12:30:00", 12.5),
    ("48:00:00", 48),
    ("4-00:00:00", 96),
    ("1-12:00:00", 36),
])
def test_parse_slurm_time(value, hours):
    assert parse_slurm_time(value) == pytest.approx(hours)


@pytest.mark.parametrize("value", ["12h", "12:00", "", "12:60:00", "nonsense"])
def test_parse_slurm_time_rejects_junk(value):
    with pytest.raises(ServingConfigError):
        parse_slurm_time(value)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_gpus_beyond_the_node_are_rejected(tmp_path):
    with pytest.raises(ServingConfigError, match="outside 1"):
        load_serving_config(write_config(tmp_path, gpus=MAX_GPUS + 1))


def test_tensor_parallel_may_not_exceed_gpus(tmp_path):
    with pytest.raises(ServingConfigError, match="exceeds gpus"):
        load_serving_config(write_config(tmp_path, gpus=2, tensor_parallel_size=4))


def test_tensor_parallel_must_divide_gpus(tmp_path):
    with pytest.raises(ServingConfigError, match="not a multiple"):
        load_serving_config(write_config(tmp_path, gpus=3, tensor_parallel_size=2))


def test_context_must_leave_room_for_the_prompt(tmp_path):
    """max_model_len covers prompt + completion, so it cannot equal max_tokens."""
    with pytest.raises(ServingConfigError, match="tokens for a prompt"):
        load_serving_config(write_config(tmp_path, max_model_len=16384))


def test_walltime_beyond_the_partition_limit_is_rejected(tmp_path):
    with pytest.raises(ServingConfigError, match="exceeds the"):
        load_serving_config(write_config(tmp_path, partition="short", time="12:00:00"))


def test_unknown_partition_is_rejected(tmp_path):
    with pytest.raises(ServingConfigError, match="unknown partition"):
        load_serving_config(write_config(tmp_path, partition="gpu"))


def test_missing_serving_key_names_the_key(tmp_path):
    serving = dict(hf_model_id="org/x", gpus=1, tensor_parallel_size=1,
                   max_model_len=20480, gpu_memory_utilization=0.9,
                   partition="verylong", time="12:00:00")
    del serving["max_model_len"]
    entry = {"model": "m", "max_tokens": 16384, "port": 8100,
             "serving": serving, "paths": {}}
    path = tmp_path / "c.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    with pytest.raises(ServingConfigError, match="max_model_len"):
        load_serving_config(path)


def test_config_without_a_serving_block_says_why(tmp_path):
    path = tmp_path / "api.json"
    path.write_text(json.dumps([{"model": "gpt-5.2", "max_tokens": 1024,
                                 "port": 8000, "paths": {}}]), encoding="utf-8")
    with pytest.raises(ServingConfigError, match="no 'serving' block"):
        load_serving_config(path)


# ---------------------------------------------------------------------------
# shell emission -- the SLURM scripts `eval` this
# ---------------------------------------------------------------------------

def test_shell_assignments_cover_what_the_scripts_read():
    emitted = make().to_shell_assignments()
    for key in ("MODEL", "JOB_NAME", "HF_MODEL_ID", "GPUS", "TENSOR_PARALLEL_SIZE",
                "MAX_MODEL_LEN", "MAX_TOKENS", "PORT", "PARTITION", "TIME_LIMIT",
                "EXTRA_ARGS", "VLLM_ARGS"):
        assert f"{key}=" in emitted, f"{key} missing from shell output"


def test_shell_assignments_quote_hostile_values():
    """A stray space or quote in a config must not become shell syntax."""
    serving = make(model="weird name; rm -rf /", hf_model_id="org/x y")
    emitted = serving.to_shell_assignments()
    model_line = next(l for l in emitted.splitlines() if l.startswith("MODEL="))
    # shlex round-trips the quoting: splitting gives back the literal value.
    assert shlex.split(model_line.removeprefix("MODEL="))[0] == "weird name; rm -rf /"


def test_vllm_args_include_the_reasoning_parser_only_when_set():
    assert "--reasoning-parser" not in make().vllm_args()
    assert make(reasoning_parser="qwen3").vllm_args()[-2:] == ["--reasoning-parser", "qwen3"]


def test_extra_args_are_appended_verbatim():
    args = make(extra_args=("--enforce-eager",)).vllm_args()
    assert args[-1] == "--enforce-eager"


# ---------------------------------------------------------------------------
# the shipped sweep
# ---------------------------------------------------------------------------

def model_size_configs():
    return sorted(MODEL_SIZE_DIR.glob("*.json"))


def test_the_sweep_is_present():
    assert len(model_size_configs()) == 12, "expected twelve model-size configs"


@pytest.mark.parametrize("path", model_size_configs(), ids=lambda p: p.stem)
def test_every_sweep_config_validates(path):
    load_serving_config(path)


@pytest.mark.parametrize("path", model_size_configs(), ids=lambda p: p.stem)
def test_every_sweep_config_is_held_constant_where_it_must_be(path):
    """Size is the only thing allowed to vary across the sweep.

    Temperature, prompt template, persona set, retry budget and reasoning mode
    are the confounds this ablation exists to exclude.
    """
    entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
    assert entry["temperature"] == 0.7
    assert entry["max_tries"] == 3
    assert entry["chat_template_kwargs"] == {"enable_thinking": True}
    assert entry["paths"]["prompt_template_file"] == "prompt_templates/templates2.txt"
    assert entry["paths"]["personas_root"] == "data/prolific_data"


@pytest.mark.parametrize("path", model_size_configs(), ids=lambda p: p.stem)
def test_sweep_results_go_to_the_model_size_folder(path):
    entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
    output = entry["paths"].get("output_excel") or entry["paths"]["disabled_output_excel"]
    assert output.startswith("results/model_size/")
    assert entry["model"] in output


def test_sweep_ports_are_unique():
    """Two configs on one port would silently talk to the wrong server."""
    ports = {}
    for path in model_size_configs():
        entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
        ports.setdefault(entry["port"], []).append(path.name)
    clashes = {p: names for p, names in ports.items() if len(names) > 1}
    assert not clashes, f"duplicate ports: {clashes}"


def test_sweep_model_names_match_their_filenames():
    for path in model_size_configs():
        entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
        assert entry["model"] == path.stem


def test_sweep_ports_do_not_collide_with_the_existing_configs():
    existing = set()
    for path in CONFIGS_DIR.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
        if "port" in entry:
            existing.add(entry["port"])
    for path in model_size_configs():
        entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
        assert entry["port"] not in existing, f"{path.name} reuses port {entry['port']}"


def test_larger_models_are_not_given_fewer_gpus():
    """A sanity check on the hand-written allocation: GPUs are monotone in size."""
    by_gpus = {}
    for path in model_size_configs():
        serving = load_serving_config(path)
        by_gpus[serving.model] = serving.gpus
    assert by_gpus["Qwen3.5-122B-A10B"] == 4
    assert by_gpus["Qwen3.5-0.8B"] <= by_gpus["Qwen3.5-27B"] <= by_gpus["Qwen3.5-122B-A10B"]
    assert by_gpus["gemma-4-E2B-it"] <= by_gpus["gemma-4-31B-it"]


# ---------------------------------------------------------------------------
# the CLI the SLURM scripts actually call
# ---------------------------------------------------------------------------

def test_cli_emits_evaluable_shell():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params",
         "model_size/Qwen3.5-9B.json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "GPUS=1" in result.stdout
    assert "VLLM_ARGS=(" in result.stdout


def test_cli_check_passes_for_the_whole_sweep():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params", "--check",
         *[str(p) for p in model_size_configs()]],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_cli_writes_errors_to_stderr_not_stdout(tmp_path):
    """`eval "$(...)"` must never evaluate an error message."""
    bad = write_config(tmp_path, gpus=99)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params", str(bad)],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "error:" in result.stderr
