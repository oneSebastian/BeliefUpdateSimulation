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


def test_pilot_all_is_chained_by_default(shim):
    """One model resident at a time, so the node stays free for other users."""
    env, log = shim
    result = run(env, "--pilot-all")
    assert result.returncode == 0, result.stderr

    submissions = calls(log)
    # The first job has nothing to wait for.
    assert "--dependency" not in submissions[0]
    # Every later job waits on the one before it.
    for i, line in enumerate(submissions[1:], start=1):
        assert "--dependency=afterany:" in line, f"job {i} was not chained: {line}"

    # The chain is consecutive: job N depends on the id returned for job N-1.
    ids = [1001 + i for i in range(len(submissions))]
    for i, line in enumerate(submissions[1:], start=1):
        assert f"--dependency=afterany:{ids[i - 1]}" in line


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


def test_only_still_chains_what_it_selects(shim):
    env, log = shim
    run(env, "--pilot-all", "--only", "gemma-4-*")
    submissions = calls(log)
    assert "--dependency" not in submissions[0]
    assert all("--dependency=afterany:" in line for line in submissions[1:])


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


def test_pilot_walltime_is_sized_per_model_for_the_weight_download(shim):
    """One shared value would either strand the 244GB fetch or over-reserve
    for the 2GB one. The big model gets more, the small model gets less."""
    env, log = shim
    run(env, "--pilot-all")
    submissions = {}
    for line in calls(log):
        model = next(f for f in line.split() if f.startswith("--job-name="))
        time = next(f for f in line.split() if f.startswith("--time="))
        submissions[model.removeprefix("--job-name=")] = time.removeprefix("--time=")

    assert submissions["Qwen3.5-122B-A10B-pilot"] == "12:00:00"
    assert submissions["Qwen3.5-0.8B-pilot"] == "02:00:00"
    assert submissions["Qwen3.5-27B-pilot"] == "06:00:00"
    # Strictly ordered: bigger download, longer reservation.
    assert (submissions["Qwen3.5-0.8B-pilot"]
            < submissions["Qwen3.5-27B-pilot"]
            < submissions["Qwen3.5-122B-A10B-pilot"])


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
