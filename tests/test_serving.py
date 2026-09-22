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


def test_pilot_time_beyond_the_partition_limit_is_rejected(tmp_path):
    with pytest.raises(ServingConfigError, match="pilot_time.*exceeds the"):
        load_serving_config(write_config(tmp_path, partition="long", time="06:00:00",
                                         pilot_time="08:00:00"))


def test_pilot_time_longer_than_the_full_run_is_rejected(tmp_path):
    """A 15-call pilot cannot need longer than 391 personas."""
    with pytest.raises(ServingConfigError, match="cannot need longer"):
        load_serving_config(write_config(tmp_path, time="04:00:00",
                                         pilot_time="08:00:00"))


def test_pilot_time_defaults_when_a_config_omits_it(tmp_path):
    from belief_update_sim.serving import DEFAULT_PILOT_TIME
    assert load_serving_config(write_config(tmp_path)).pilot_time == DEFAULT_PILOT_TIME


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
                "PILOT_TIME", "EXTRA_ARGS", "VLLM_ARGS"):
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

    Temperature, prompt template, persona set, retry budget, reasoning mode and
    token budget are the confounds this ablation exists to exclude.
    """
    entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
    assert entry["temperature"] == 0.7
    assert entry["max_tries"] == 3
    assert entry["chat_template_kwargs"] == {"enable_thinking": True}
    assert entry["paths"]["prompt_template_file"] == "prompt_templates/templates2.txt"
    assert entry["paths"]["personas_root"] == "data/prolific_data"


def test_the_token_budget_is_uniform_across_the_sweep():
    """It was tiered (16k small / 32k large), which gave the small models less
    room than the large ones -- budget confounded with size, in a sweep whose
    premise is that only size varies. A model that answers stops when it is
    done, so a uniform ceiling costs the healthy models nothing.
    """
    budgets = {s.max_tokens for s in map(load_serving_config, model_size_configs())}
    contexts = {s.max_model_len for s in map(load_serving_config, model_size_configs())}
    assert budgets == {32768}, f"token budget varies across the sweep: {budgets}"
    assert len(contexts) == 1, f"context length varies across the sweep: {contexts}"


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


@pytest.mark.parametrize("path", model_size_configs(), ids=lambda p: p.stem)
def test_every_sweep_config_sizes_its_own_pilot_walltime(path):
    """Left to the default, the 244GB download would hit the walltime."""
    entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
    assert "pilot_time" in entry["serving"], (
        f"{path.name}: declare pilot_time so a cold weight download fits"
    )


def test_pilot_walltime_is_monotone_in_model_size():
    """Bigger weights mean a longer download, so a longer pilot reservation."""
    pilot = {s.model: parse_slurm_time(s.pilot_time)
             for s in map(load_serving_config, model_size_configs())}
    assert pilot["Qwen3.5-0.8B"] <= pilot["Qwen3.5-9B"] <= pilot["Qwen3.5-27B"] \
        <= pilot["Qwen3.5-122B-A10B"]
    assert pilot["gemma-4-E2B-it"] <= pilot["gemma-4-31B-it"]
    # The largest download in the sweep (~244GB) gets the longest slot.
    assert pilot["Qwen3.5-122B-A10B"] == max(pilot.values())


@pytest.mark.parametrize("path", model_size_configs(), ids=lambda p: p.stem)
def test_pilot_walltime_is_well_under_the_full_run(path):
    serving = load_serving_config(path)
    assert parse_slurm_time(serving.pilot_time) < parse_slurm_time(serving.time_limit)


def test_every_sweep_config_sets_a_reasoning_parser():
    """Thinking is forced on for all twelve, so every model needs the parser
    that splits it out of the answer -- qwen3 for Qwen, gemma4 for Gemma."""
    expected = {"Qwen3.5": "qwen3", "gemma-4": "gemma4"}
    for path in model_size_configs():
        serving = load_serving_config(path)
        family = next(f for f in expected if serving.model.startswith(f))
        assert serving.reasoning_parser == expected[family], (
            f"{path.name}: expected {expected[family]!r}, "
            f"got {serving.reasoning_parser!r}"
        )


def test_qwen_configs_avoid_the_flashinfer_gdn_jit():
    """Qwen3.5's gated-delta-net prefill kernel is JIT-compiled with nvcc, and
    the cluster's /usr/bin/nvcc cannot target sm_90a. Triton needs no nvcc."""
    for path in model_size_configs():
        serving = load_serving_config(path)
        if not serving.model.startswith("Qwen3.5"):
            continue
        args = serving.vllm_args()
        assert "--gdn-prefill-backend" in args, f"{path.name}: GDN JIT not disabled"
        assert args[args.index("--gdn-prefill-backend") + 1] == "triton"


def test_gemma_configs_do_not_carry_the_gdn_flag():
    """Gemma 4 has no gated-delta-net layers; the flag would be meaningless."""
    for path in model_size_configs():
        serving = load_serving_config(path)
        if serving.model.startswith("gemma-4"):
            assert "--gdn-prefill-backend" not in serving.vllm_args()


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


def test_sort_key_orders_by_gpus_before_anything_else():
    from scripts.pipeline.serving_params import sort_key
    small_but_many_gpus = make(gpus=4, tensor_parallel_size=4, pilot_time="02:00:00")
    large_but_one_gpu = make(gpus=1, tensor_parallel_size=1, pilot_time="12:00:00",
                             time_limit="48:00:00")
    assert sort_key(large_but_one_gpu) < sort_key(small_but_many_gpus)


def test_sort_key_breaks_gpu_ties_by_download_size():
    from scripts.pipeline.serving_params import sort_key
    small = make(model="a-small", pilot_time="02:00:00")
    big = make(model="z-big", pilot_time="04:00:00")
    assert sort_key(small) < sort_key(big)
    # ...and the name only matters once both of those are equal.
    first = make(model="aaa", pilot_time="02:00:00")
    second = make(model="zzz", pilot_time="02:00:00")
    assert sort_key(first) < sort_key(second)


def test_waves_never_exceed_the_gpu_capacity():
    from scripts.pipeline.serving_params import plan_waves
    servings = sorted((load_serving_config(p) for p in model_size_configs()),
                      key=lambda s: s.gpus)
    for wave in plan_waves(servings, 4):
        assert sum(s.gpus for s in wave) <= 4


def test_every_job_lands_in_exactly_one_wave():
    from scripts.pipeline.serving_params import plan_waves
    servings = [load_serving_config(p) for p in model_size_configs()]
    placed = [s.model for wave in plan_waves(servings, 4) for s in wave]
    assert sorted(placed) == sorted(s.model for s in servings)
    assert len(placed) == len(set(placed))


def test_waves_are_packed_not_merely_split():
    """Four 1-GPU jobs must share a wave, not take four of them."""
    from scripts.pipeline.serving_params import plan_waves
    ones = [make(model=f"m{i}") for i in range(4)]
    assert len(plan_waves(ones, 4)) == 1


def test_capacity_one_is_strictly_sequential():
    """--sequential is expressed as --max-gpus 1; every wave holds one job."""
    from scripts.pipeline.serving_params import plan_waves
    servings = [load_serving_config(p) for p in model_size_configs()]
    waves = plan_waves(servings, 1)
    assert len(waves) == len(servings)
    assert all(len(w) == 1 for w in waves)


def test_a_job_larger_than_the_capacity_gets_its_own_wave():
    """Otherwise the 4-GPU model could never be scheduled at all."""
    from scripts.pipeline.serving_params import plan_waves
    big = make(model="big", gpus=4, tensor_parallel_size=4)
    small = make(model="small", gpus=1)
    waves = plan_waves([small, big], 2)
    assert [ [s.model for s in w] for w in waves ] == [["small"], ["big"]]


def test_the_sweep_packs_into_the_expected_waves():
    from scripts.pipeline.serving_params import plan_waves, sort_key
    servings = sorted((load_serving_config(p) for p in model_size_configs()),
                      key=sort_key)
    waves = plan_waves(servings, 4)
    sizes = [sum(s.gpus for s in w) for w in waves]
    assert sizes == [4, 3, 4, 4, 4], sizes
    # The 4-GPU model is alone in the last wave.
    assert [s.model for s in waves[-1]] == ["Qwen3.5-122B-A10B"]


def test_cli_plan_waves_emits_wave_and_config():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params", "--plan-waves",
         *[str(p) for p in model_size_configs()]],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    lines = [l.split(None, 1) for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 12
    waves = [int(w) for w, _ in lines]
    assert waves == sorted(waves), "waves must be emitted in order"
    assert set(waves) == {1, 2, 3, 4, 5}


def test_cli_plan_waves_rejects_a_zero_capacity():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params", "--plan-waves",
         "--max-gpus", "0", *[str(p) for p in model_size_configs()]],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 1
    assert result.stdout == ""


def test_cli_sort_by_gpus_lists_the_sweep_cheapest_first():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pipeline.serving_params", "--sort-by-gpus",
         *[str(p) for p in model_size_configs()]],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    ordered = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(ordered) == 12
    gpus = [load_serving_config(p).gpus for p in ordered]
    assert gpus == sorted(gpus), gpus
    assert "Qwen3.5-122B-A10B" in ordered[-1]


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
