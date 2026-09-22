"""Cluster-serving parameters, read from the same JSON config as the agent run.

The model-size sweep runs twelve locally served models that differ in how many
GPUs they need, how long they take, and how much context they want. Encoding
that in twelve copies of a SLURM script is how the existing ``run_qwen32.slurm``
and ``run_Llama70b.slurm`` diverged; instead every parameter lives in the
config's ``serving`` block and one generic job script reads it.

``#SBATCH --gpus=`` is fixed at *submit* time, not at run time, so the values
have to be readable from the login node before ``sbatch`` is called. That is
what :func:`to_shell_assignments` is for: ``slurm/submit_model.sh`` evaluates
its output, passes ``--gpus``/``--time``/``--partition`` to ``sbatch``, and
``slurm/run_model.slurm`` evaluates it again to build the ``vllm serve``
command.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIGS_DIR

# The cluster is a single node (labgpu01) with four H100 80GB cards.
MAX_GPUS = 4
GPU_MEMORY_GB = 80

# vLLM shards attention heads across tensor-parallel ranks, and head counts are
# powers of two in every model in this sweep. Anything else is a typo.
VALID_TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)

# Partition time limits, from `sinfo`. A job asking for more than its
# partition allows is rejected at submit time, which is a slow way to find a
# typo -- so check it here instead.
PARTITION_TIME_LIMITS_HOURS = {
    "short": 1,
    "medium": 3,
    "long": 6,
    "verylong": 4 * 24,
    "ultralong": 28 * 24,
}

# The assembled prompt (template2 + demographics + Big-5 + three comments) runs
# to roughly 2.2k tokens. `max_model_len` has to cover prompt *and* completion,
# so it must exceed `max_tokens` by at least this much or long personas will be
# rejected by the server rather than merely truncated.
PROMPT_HEADROOM_TOKENS = 4096

_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d{1,3}):([0-5]\d):([0-5]\d)$")

REQUIRED_SERVING_KEYS = (
    "hf_model_id",
    "gpus",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "partition",
    "time",
)


class ServingConfigError(ValueError):
    """A config's `serving` block is missing or internally inconsistent."""


def parse_slurm_time(value: str) -> float:
    """SLURM walltime (``HH:MM:SS`` or ``D-HH:MM:SS``) as hours."""
    match = _TIME_RE.match(value)
    if match is None:
        raise ServingConfigError(
            f"time {value!r} is not a SLURM walltime (HH:MM:SS or D-HH:MM:SS)"
        )
    days, hours, minutes, seconds = match.groups()
    return (
        int(days or 0) * 24
        + int(hours)
        + int(minutes) / 60
        + int(seconds) / 3600
    )


@dataclass(frozen=True)
class ServingConfig:
    """Everything the SLURM job and the vLLM server need, validated."""

    config_name: str
    model: str
    hf_model_id: str
    gpus: int
    tensor_parallel_size: int
    max_model_len: int
    max_tokens: int
    gpu_memory_utilization: float
    port: int
    partition: str
    time_limit: str
    reasoning_parser: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def job_name(self) -> str:
        return self.model

    @property
    def weights_gb_budget(self) -> float:
        """GPU memory vLLM is allowed to claim, across all ranks."""
        return self.gpus * GPU_MEMORY_GB * self.gpu_memory_utilization

    def vllm_args(self) -> list[str]:
        """The `vllm serve` argument vector, excluding the executable."""
        args = [
            self.hf_model_id,
            "--served-model-name", self.model,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--port", str(self.port),
        ]
        if self.reasoning_parser:
            args += ["--reasoning-parser", self.reasoning_parser]
        args += list(self.extra_args)
        return args

    def to_shell_assignments(self) -> str:
        """Shell assignments for `eval` in the SLURM scripts.

        Every value is quoted with shlex, and `EXTRA_ARGS` / `VLLM_ARGS` are
        emitted as bash arrays so that arguments containing spaces survive.
        """
        def q(value):
            return shlex.quote(str(value))

        lines = [
            f"MODEL={q(self.model)}",
            f"CONFIG_NAME={q(self.config_name)}",
            f"JOB_NAME={q(self.job_name)}",
            f"HF_MODEL_ID={q(self.hf_model_id)}",
            f"GPUS={q(self.gpus)}",
            f"TENSOR_PARALLEL_SIZE={q(self.tensor_parallel_size)}",
            f"MAX_MODEL_LEN={q(self.max_model_len)}",
            f"MAX_TOKENS={q(self.max_tokens)}",
            f"GPU_MEMORY_UTILIZATION={q(self.gpu_memory_utilization)}",
            f"PORT={q(self.port)}",
            f"PARTITION={q(self.partition)}",
            f"TIME_LIMIT={q(self.time_limit)}",
            f"REASONING_PARSER={q(self.reasoning_parser or '')}",
            "EXTRA_ARGS=({})".format(" ".join(q(a) for a in self.extra_args)),
            "VLLM_ARGS=({})".format(" ".join(q(a) for a in self.vllm_args())),
        ]
        return "\n".join(lines)


def _require(mapping, key, config_name, where):
    if key not in mapping:
        raise ServingConfigError(f"{config_name}: {where} is missing {key!r}")
    return mapping[key]


def _validate(serving: ServingConfig) -> ServingConfig:
    name = serving.config_name

    if not 1 <= serving.gpus <= MAX_GPUS:
        raise ServingConfigError(
            f"{name}: gpus={serving.gpus} outside 1..{MAX_GPUS} "
            f"(the cluster has {MAX_GPUS} H100s on one node)"
        )

    if serving.tensor_parallel_size not in VALID_TENSOR_PARALLEL_SIZES:
        raise ServingConfigError(
            f"{name}: tensor_parallel_size={serving.tensor_parallel_size} "
            f"is not one of {VALID_TENSOR_PARALLEL_SIZES}"
        )

    if serving.tensor_parallel_size > serving.gpus:
        raise ServingConfigError(
            f"{name}: tensor_parallel_size={serving.tensor_parallel_size} "
            f"exceeds gpus={serving.gpus}"
        )

    if serving.gpus % serving.tensor_parallel_size != 0:
        raise ServingConfigError(
            f"{name}: gpus={serving.gpus} is not a multiple of "
            f"tensor_parallel_size={serving.tensor_parallel_size}"
        )

    if not 0 < serving.gpu_memory_utilization <= 1:
        raise ServingConfigError(
            f"{name}: gpu_memory_utilization="
            f"{serving.gpu_memory_utilization} outside (0, 1]"
        )

    if serving.max_model_len < serving.max_tokens + PROMPT_HEADROOM_TOKENS:
        raise ServingConfigError(
            f"{name}: max_model_len={serving.max_model_len} leaves "
            f"{serving.max_model_len - serving.max_tokens} tokens for a prompt "
            f"that needs ~{PROMPT_HEADROOM_TOKENS}; "
            f"raise it to at least {serving.max_tokens + PROMPT_HEADROOM_TOKENS}"
        )

    if serving.partition not in PARTITION_TIME_LIMITS_HOURS:
        raise ServingConfigError(
            f"{name}: unknown partition {serving.partition!r}; "
            f"expected one of {sorted(PARTITION_TIME_LIMITS_HOURS)}"
        )

    requested = parse_slurm_time(serving.time_limit)
    allowed = PARTITION_TIME_LIMITS_HOURS[serving.partition]
    if requested > allowed:
        raise ServingConfigError(
            f"{name}: time={serving.time_limit} ({requested:.2f}h) exceeds the "
            f"{allowed}h limit of partition {serving.partition!r}"
        )

    if not 1024 <= serving.port <= 65535:
        raise ServingConfigError(f"{name}: port={serving.port} outside 1024..65535")

    return serving


def load_config_entry(config: str | Path) -> dict:
    """Load a config by name (resolved under ``configs/``) or by path.

    The shipped configs carry a UTF-8 BOM, hence ``utf-8-sig``.
    """
    path = Path(config)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / config
    if not path.exists():
        raise ServingConfigError(f"config not found: {config}")

    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ServingConfigError(f"{path.name}: expected a non-empty JSON list")
    return entries[0]


def load_serving_config(config: str | Path) -> ServingConfig:
    """Parse and validate the `serving` block of a config."""
    path = Path(config)
    name = path.name if path.suffix == ".json" else str(config)
    entry = load_config_entry(config)

    if "serving" not in entry:
        raise ServingConfigError(
            f"{name}: no 'serving' block. Only configs that are served locally "
            f"on the cluster carry one; API-backed models are run directly with "
            f"`python -m scripts.pipeline.run_agent --config {name}`."
        )

    serving = entry["serving"]
    for key in REQUIRED_SERVING_KEYS:
        _require(serving, key, name, "serving")

    return _validate(ServingConfig(
        config_name=name,
        model=_require(entry, "model", name, "config"),
        hf_model_id=serving["hf_model_id"],
        gpus=int(serving["gpus"]),
        tensor_parallel_size=int(serving["tensor_parallel_size"]),
        max_model_len=int(serving["max_model_len"]),
        max_tokens=int(_require(entry, "max_tokens", name, "config")),
        gpu_memory_utilization=float(serving["gpu_memory_utilization"]),
        port=int(_require(entry, "port", name, "config")),
        partition=serving["partition"],
        time_limit=serving["time"],
        reasoning_parser=serving.get("reasoning_parser") or None,
        extra_args=tuple(serving.get("extra_args", ())),
    ))
