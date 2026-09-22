"""Tests for slurm/submit_model.sh, driven against a shimmed `sbatch`.

The dependency chain is the part worth pinning: if it silently stops being
emitted, twelve jobs land in the queue at once and take the whole node. Nothing
here submits anything -- `sbatch` is replaced by a script that records its
argv and prints a fake job id.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from belief_update_sim.config import PROJECT_ROOT

SUBMIT = PROJECT_ROOT / "slurm" / "submit_model.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


@pytest.fixture
def shim(tmp_path):
    """A fake `sbatch` on PATH that logs argv and returns increasing job ids."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    counter = tmp_path / "counter"
    counter.write_text("1000")

    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{log.as_posix()}"\n'
        f'n=$(cat "{counter.as_posix()}")\n'
        f'n=$((n + 1)); echo "$n" > "{counter.as_posix()}"\n'
        'echo "$n"\n'
    )
    sbatch.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bindir.as_posix()}{os.pathsep}{env['PATH']}"
    env["SKIP_CONDA"] = "1"
    env["PYTHON"] = sys.executable
    return env, log


def run(env, *args):
    result = subprocess.run(
        ["bash", str(SUBMIT), *args],
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=env,
    )
    return result


def calls(log: Path):
    if not log.exists():
        return []
    return [line for line in log.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# batch submission
# ---------------------------------------------------------------------------

def test_pilot_all_submits_every_config(shim):
    env, log = shim
    result = run(env, "--pilot-all")
    assert result.returncode == 0, result.stderr
    assert len(calls(log)) == 12


def parse_submissions(log: Path):
    """[(jobid, gpus, dependency_ids)] in submission order."""
    out = []
    for i, line in enumerate(calls(log)):
        fields = line.split()
        gpus = int(next(f for f in fields if f.startswith("--gpus=")).removeprefix("--gpus="))
        dep = next((f for f in fields if f.startswith("--dependency=")), None)
        deps = []
        if dep:
            spec = dep.removeprefix("--dependency=")
            assert spec.startswith("afterany:"), spec
            deps = spec.removeprefix("afterany:").split(":")
        out.append((str(1001 + i), gpus, deps))
    return out


def waves_from(submissions):
    """Group submissions by their dependency set -- that is what a wave is."""
    waves, current, key = [], [], None
    for jobid, gpus, deps in submissions:
        if tuple(deps) != key:
            if current:
                waves.append(current)
            current, key = [], tuple(deps)
        current.append((jobid, gpus, deps))
    if current:
        waves.append(current)
    return waves


def test_pilot_all_is_packed_into_waves(shim):
    """Up to 4 GPUs at once; the rest wait on a dependency, not on resources."""
    env, log = shim
    result = run(env, "--pilot-all")
    assert result.returncode == 0, result.stderr

    submissions = parse_submissions(log)
    assert len(submissions) == 12

    waves = waves_from(submissions)
    # No wave asks for more than the node has.
    for wave in waves:
        assert sum(gpus for _, gpus, _ in wave) <= 4, wave
    # More than one job runs concurrently -- this is not a serial chain.
    assert any(len(w) > 1 for w in waves)


def test_the_first_wave_has_no_dependency(shim):
    env, log = shim
    run(env, "--pilot-all")
    waves = waves_from(parse_submissions(log))
    assert all(deps == [] for _, _, deps in waves[0])
    assert len(waves[0]) == 4, "four 1-GPU jobs should share the first wave"


def test_each_wave_depends_on_every_job_of_the_previous_one(shim):
    """Not just the last one: the wave is only clear when all of it is done."""
    env, log = shim
    run(env, "--pilot-all")
    waves = waves_from(parse_submissions(log))

    for previous, wave in zip(waves, waves[1:]):
        expected = [jobid for jobid, _, _ in previous]
        for jobid, _, deps in wave:
            assert deps == expected, f"job {jobid}: {deps} != {expected}"


def test_every_job_after_the_first_wave_is_held_by_a_dependency(shim):
    """The point of the whole design: a job pending on a dependency requests
    no resources, so it cannot take a backfill reservation from another user."""
    env, log = shim
    run(env, "--pilot-all")
    waves = waves_from(parse_submissions(log))
    for wave in waves[1:]:
        for jobid, _, deps in wave:
            assert deps, f"job {jobid} would queue for resources instead of waiting"


def test_max_gpus_one_restores_the_serial_chain(shim):
    env, log = shim
    result = run(env, "--pilot-all", "--max-gpus", "1")
    assert result.returncode == 0, result.stderr
    waves = waves_from(parse_submissions(log))
    assert len(waves) == 12
    assert all(len(w) == 1 for w in waves)


def test_sequential_is_the_same_as_max_gpus_one(shim):
    env, log = shim
    run(env, "--pilot-all", "--sequential")
    assert all(len(w) == 1 for w in waves_from(parse_submissions(log)))


def test_max_gpus_two_packs_pairs_of_single_gpu_jobs(shim):
    """A wave may only exceed the cap when one job alone does.

    Qwen3.5-122B-A10B needs 4 GPUs; --max-gpus 2 cannot shrink that, so it
    takes a solo wave rather than being dropped from the sweep.
    """
    env, log = shim
    run(env, "--pilot-all", "--max-gpus", "2")
    oversized = 0
    for wave in waves_from(parse_submissions(log)):
        total = sum(gpus for _, gpus, _ in wave)
        if total > 2:
            assert len(wave) == 1, f"packed past the cap: {wave}"
            oversized += 1
        # Pairs of 1-GPU jobs must actually share a wave.
    assert oversized == 1, "only the 4-GPU model should exceed a cap of 2"
    assert any(len(w) == 2 for w in waves_from(parse_submissions(log)))


def test_max_gpus_without_a_number_is_rejected(shim):
    env, _ = shim
    result = run(env, "--pilot-all", "--max-gpus")
    assert result.returncode == 2
    assert "--max-gpus needs a number" in result.stderr


def test_afterany_not_afterok_so_one_failure_does_not_strand_the_rest(shim):
    env, log = shim
    run(env, "--pilot-all")
    joined = "\n".join(calls(log))
    assert "afterany" in joined
    assert "afterok" not in joined


def test_parallel_opts_out_of_the_chain(shim):
    env, log = shim
    result = run(env, "--pilot-all", "--parallel")
    assert result.returncode == 0, result.stderr
    assert all("--dependency" not in line for line in calls(log))


def test_after_chains_the_first_job_behind_an_existing_one(shim):
    env, log = shim
    result = run(env, "--pilot-all", "--after", "999")
    assert result.returncode == 0, result.stderr
    assert "--dependency=afterany:999" in calls(log)[0]


def test_only_restricts_the_batch_to_matching_configs(shim):
    """For re-running one family after a fix, without redoing the rest."""
    env, log = shim
    result = run(env, "--pilot-all", "--only", "gemma-4-*")
    assert result.returncode == 0, result.stderr
    submissions = calls(log)
    assert len(submissions) == 5
    assert all("gemma-4-" in line for line in submissions)
    assert not any("Qwen" in line for line in submissions)


def test_only_still_packs_what_it_selects(shim):
    """The five Gemma models pack into 1+1+1 then 2+2, not a serial chain."""
    env, log = shim
    run(env, "--pilot-all", "--only", "gemma-4-*")
    waves = waves_from(parse_submissions(log))

    assert [sum(g for _, g, _ in w) for w in waves] == [3, 4]
    assert all(deps == [] for _, _, deps in waves[0])
    for previous, wave in zip(waves, waves[1:]):
        expected = [jobid for jobid, _, _ in previous]
        assert all(deps == expected for _, _, deps in wave)


def test_only_matching_nothing_fails_loudly(shim):
    env, log = shim
    result = run(env, "--pilot-all", "--only", "llama-*")
    assert result.returncode == 1
    assert "no configs" in result.stderr
    assert calls(log) == []


def test_only_without_a_glob_is_rejected(shim):
    env, _ = shim
    result = run(env, "--pilot-all", "--only")
    assert result.returncode == 2
    assert "--only needs a glob" in result.stderr


def test_after_without_a_job_id_is_rejected(shim):
    env, _ = shim
    result = run(env, "--pilot-all", "--after")
    assert result.returncode == 2
    assert "--after needs a job id" in result.stderr


# ---------------------------------------------------------------------------
# what reaches sbatch and run_agent
# ---------------------------------------------------------------------------

def test_pilot_all_passes_the_pilot_flag_to_the_agent(shim):
    env, log = shim
    run(env, "--pilot-all")
    assert all("--pilot" in line for line in calls(log))


def test_all_mode_does_not_pass_the_pilot_flag(shim):
    env, log = shim
    run(env, "--all")
    assert all("--pilot" not in line for line in calls(log))


def test_passthrough_args_reach_the_agent(shim):
    env, log = shim
    run(env, "--all", "--resume")
    assert all(line.rstrip().endswith("--resume") for line in calls(log))


def test_pilots_do_not_reserve_the_full_run_walltime(shim):
    """15 calls must not hold a 48h slot -- it blocks the node and backfill."""
    env, log = shim
    run(env, "--pilot-all")
    for line in calls(log):
        assert "--time=48:00:00" not in line, line
        assert "--time=24:00:00" not in line, line


def submitted_walltimes(log: Path):
    """job name -> the --time sbatch was given."""
    out = {}
    for line in calls(log):
        fields = line.split()
        name = next(f for f in fields if f.startswith("--job-name="))
        time = next(f for f in fields if f.startswith("--time="))
        out[name.removeprefix("--job-name=")] = time.removeprefix("--time=")
    return out


def test_pilot_walltime_comes_from_the_config(shim):
    """The value itself is a tuning knob and lives in the config; what must
    hold is that submit passes each model's own pilot_time, not a shared one."""
    from belief_update_sim.serving import load_serving_config

    env, log = shim
    run(env, "--pilot-all")
    submitted = submitted_walltimes(log)

    for path in sorted((PROJECT_ROOT / "configs" / "model_size").glob("*.json")):
        serving = load_serving_config(path)
        assert submitted[f"{serving.model}-pilot"] == serving.pilot_time


def test_pilot_walltime_is_sized_per_model_for_the_weight_download(shim):
    """One shared value would either strand the 244GB fetch or over-reserve
    for the 2GB one. The big model gets more, the small model gets less."""
    from belief_update_sim.serving import parse_slurm_time

    env, log = shim
    run(env, "--pilot-all")
    submitted = submitted_walltimes(log)
    hours = {name: parse_slurm_time(t) for name, t in submitted.items()}

    assert (hours["Qwen3.5-0.8B-pilot"]
            < hours["Qwen3.5-27B-pilot"]
            < hours["Qwen3.5-122B-A10B-pilot"])
    assert hours["Qwen3.5-122B-A10B-pilot"] == max(hours.values())


def test_pilot_walltime_is_overridable_for_a_slow_link(shim):
    """PILOT_TIME_LIMIT overrides every model at once."""
    env, log = shim
    env = dict(env, PILOT_TIME_LIMIT="24:00:00")
    run(env, "--pilot-all")
    assert all("--time=24:00:00" in line for line in calls(log))


def test_full_runs_keep_their_configured_walltime(shim):
    env, log = shim
    run(env, "--all")
    joined = "\n".join(calls(log))
    assert "--time=48:00:00" in joined   # 122B-A10B
    assert "--time=12:00:00" in joined   # the 1-GPU models


def test_allocation_flags_come_from_the_config(shim):
    env, log = shim
    run(env, "--pilot-all")
    joined = "\n".join(calls(log))
    # The 122B model is the only 4-GPU entry in the sweep.
    assert "--gpus=4" in joined
    assert "--gpus=1" in joined
    assert "--partition=verylong" in joined


def test_single_config_submission(shim):
    env, log = shim
    result = run(env, "model_size/Qwen3.5-9B.json", "--pilot")
    assert result.returncode == 0, result.stderr
    submissions = calls(log)
    assert len(submissions) == 1
    assert "--gpus=1" in submissions[0]
    assert "Qwen3.5-9B-pilot" in submissions[0]
    assert "--dependency" not in submissions[0]


def gpus_in_submission_order(log: Path):
    counts = []
    for line in calls(log):
        field = next(f for f in line.split() if f.startswith("--gpus="))
        counts.append(int(field.removeprefix("--gpus=")))
    return counts


def test_batches_are_submitted_cheapest_allocation_first(shim):
    """The 4-GPU job can sit pending while other users hold cards. Submitting
    it last means the eleven ahead of it finish rather than stalling the chain."""
    env, log = shim
    result = run(env, "--pilot-all")
    assert result.returncode == 0, result.stderr

    counts = gpus_in_submission_order(log)
    assert counts == sorted(counts), f"not ordered by GPU count: {counts}"
    assert counts[0] == 1
    assert counts[-1] == 4


def test_the_four_gpu_model_is_submitted_last(shim):
    env, log = shim
    run(env, "--pilot-all")
    assert "Qwen3.5-122B-A10B" in calls(log)[-1]


def test_ordering_also_applies_to_full_runs(shim):
    env, log = shim
    run(env, "--all")
    counts = gpus_in_submission_order(log)
    assert counts == sorted(counts), counts


def test_ordering_survives_the_only_filter(shim):
    env, log = shim
    run(env, "--pilot-all", "--only", "Qwen3.5-*")
    counts = gpus_in_submission_order(log)
    assert counts == sorted(counts), counts
    assert "Qwen3.5-122B-A10B" in calls(log)[-1]


def test_ordering_is_stable_across_runs(shim, tmp_path):
    """Same input, same order -- otherwise a resubmission is not reproducible."""
    env, log = shim
    run(env, "--pilot-all")
    first = [line.split("--job-name=")[1].split()[0] for line in calls(log)]
    log.unlink()
    run(env, "--pilot-all")
    second = [line.split("--job-name=")[1].split()[0] for line in calls(log)]
    assert first == second


def test_run_model_disables_the_flashinfer_sampler_by_default():
    """Set in the job script, not per config, so every model in the sweep
    samples through the same kernel -- otherwise it would be a confound."""
    script = (PROJECT_ROOT / "slurm" / "run_model.slurm").read_text()
    assert 'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"' in script


def test_run_model_disables_the_flashinfer_allreduce_by_default():
    """Without this every tensor-parallel model in the sweep dies in startup:
    vLLM 0.30 dispatches all-reduce through FlashInfer, whose fused
    all-reduce+RMSNorm JIT-compiles a CUDA kernel, and the cluster's nvcc is too
    old to build it. It cost the first pilot wave all five multi-GPU models.
    """
    script = (PROJECT_ROOT / "slurm" / "run_model.slurm").read_text()
    assert 'export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"' in script


def test_both_flashinfer_switches_stay_overridable():
    """`${VAR:-0}` rather than a bare 0, so a future cluster with a newer nvcc
    can re-enable either one from the environment without editing the script."""
    script = (PROJECT_ROOT / "slurm" / "run_model.slurm").read_text()
    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLREDUCE_USE_FLASHINFER"):
        assert f'export {var}="${{{var}:-0}}"' in script


def test_the_memory_request_reaches_sbatch(shim):
    """122B-A10B was OOM-killed on its first pilot: four ranks prefetching a
    233 GiB checkpoint into a page cache charged to the job's cgroup."""
    env, log = shim
    run(env, "model_size/Qwen3.5-122B-A10B.json", "--pilot")
    assert "--mem=500G" in calls(log)[0]


def test_configs_without_a_memory_request_get_no_mem_flag(shim):
    """Requesting memory a config did not ask for would make jobs pend behind
    each other for no reason."""
    env, log = shim
    run(env, "model_size/Qwen3.5-9B.json", "--pilot")
    assert "--mem" not in calls(log)[0]


def test_a_memory_request_does_not_leak_into_the_next_job(shim):
    """The submitter evals the serving params per config inside one loop, so a
    stale MEM from the previous config would silently oversubscribe."""
    env, log = shim
    run(env, "--pilot-all")
    with_mem = [line for line in calls(log) if "--mem" in line]
    assert len(with_mem) == 1
    assert "Qwen3.5-122B-A10B" in with_mem[0]


def test_no_arguments_prints_usage_and_fails(shim):
    env, log = shim
    result = run(env)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert calls(log) == []


def test_a_bad_config_stops_before_submitting(shim):
    env, log = shim
    result = run(env, "model_size/does-not-exist.json")
    assert result.returncode != 0
    assert calls(log) == []
