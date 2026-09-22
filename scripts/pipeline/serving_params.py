"""Print a config's serving parameters, for the SLURM scripts to consume.

    eval "$(python -m scripts.pipeline.serving_params model_size/Qwen3.5-9B.json)"
    echo "$GPUS $TENSOR_PARALLEL_SIZE"
    vllm serve "${VLLM_ARGS[@]}"

`--format sh` (the default) emits shell assignments; `--format human` prints a
readable summary, which is the quickest way to sanity-check a new config before
submitting anything. `--check` validates without printing, so a whole directory
can be linted in one command:

    python -m scripts.pipeline.serving_params --check configs/model_size/*.json
"""

import argparse
import sys

from belief_update_sim.serving import (
    GPU_MEMORY_GB,
    MAX_GPUS,
    ServingConfigError,
    load_serving_config,
    parse_slurm_time,
)


def plan_waves(servings, capacity):
    """Greedily pack into waves that each fit inside `capacity` GPUs.

    Jobs within a wave run concurrently; a wave waits on the whole of the one
    before it. Expressing the batch this way rather than as one long chain
    keeps the node busy without ever requesting more than `capacity` GPUs --
    and because the waiting jobs are held by a *dependency*, they request no
    resources at all, so they cannot take a backfill reservation away from
    another user. A job needing more than `capacity` gets a wave to itself
    rather than being dropped.
    """
    waves, current, used = [], [], 0
    for serving in servings:
        if current and used + serving.gpus > capacity:
            waves.append(current)
            current, used = [], 0
        current.append(serving)
        used += serving.gpus
    if current:
        waves.append(current)
    return waves


def sort_key(serving):
    """Cheapest allocation first, so a chained batch starts immediately.

    Primary key is GPU count: the four-GPU job may sit pending while other
    users hold cards, and putting it last means the eleven ahead of it have
    already produced results by the time it waits. Within a GPU tier, pilot_time
    orders by weight-download size (it is set from the weights), so the tier
    runs roughly smallest model first; the name breaks remaining ties so the
    order is stable across runs.
    """
    return (serving.gpus, parse_slurm_time(serving.pilot_time), serving.model)


def human_summary(serving) -> str:
    lines = [
        f"config                 {serving.config_name}",
        f"model (served name)    {serving.model}",
        f"huggingface id         {serving.hf_model_id}",
        f"gpus                   {serving.gpus} x H100 {GPU_MEMORY_GB}GB "
        f"(tensor-parallel {serving.tensor_parallel_size})",
        f"gpu memory budget      {serving.weights_gb_budget:.0f} GB "
        f"(utilization {serving.gpu_memory_utilization})",
        f"context / completion   {serving.max_model_len} / {serving.max_tokens} tokens",
        f"partition / time       {serving.partition} / {serving.time_limit}",
        f"pilot walltime         {serving.pilot_time}",
        f"port                   {serving.port}",
        f"reasoning parser       {serving.reasoning_parser or '(none)'}",
        f"vllm serve             {' '.join(serving.vllm_args())}",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="+",
        help="config name under configs/ (e.g. model_size/Qwen3.5-9B.json) or a path",
    )
    parser.add_argument("--format", choices=("sh", "human"), default="sh")
    parser.add_argument(
        "--check", action="store_true",
        help="validate only; print nothing on success, exit 1 on the first error",
    )
    parser.add_argument(
        "--sort-by-gpus", action="store_true",
        help="print the given configs one per line, cheapest allocation first",
    )
    parser.add_argument(
        "--plan-waves", action="store_true",
        help="print '<wave> <config>' per line, packed into waves of --max-gpus",
    )
    parser.add_argument(
        "--max-gpus", type=int, default=MAX_GPUS, metavar="N",
        help=f"GPUs a single wave may use (default {MAX_GPUS}; 1 = strictly sequential)",
    )
    args = parser.parse_args(argv)

    batch = args.check or args.sort_by_gpus or args.plan_waves
    if len(args.config) > 1 and not batch:
        parser.error("multiple configs need --check/--sort-by-gpus/--plan-waves")

    if args.sort_by_gpus or args.plan_waves:
        if args.max_gpus < 1:
            print("error: --max-gpus must be at least 1", file=sys.stderr)
            return 1
        try:
            servings = [(c, load_serving_config(c)) for c in args.config]
        except ServingConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        ordered = sorted(servings, key=lambda pair: sort_key(pair[1]))

        if args.sort_by_gpus:
            for config, _ in ordered:
                print(config)
            return 0

        by_model = {serving.model: config for config, serving in ordered}
        for index, wave in enumerate(plan_waves([s for _, s in ordered], args.max_gpus), 1):
            for serving in wave:
                print(f"{index} {by_model[serving.model]}")
        return 0

    for config in args.config:
        try:
            serving = load_serving_config(config)
        except ServingConfigError as exc:
            # stderr, so that `eval "$(... )"` never evaluates an error message.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.check:
            continue
        print(human_summary(serving) if args.format == "human"
              else serving.to_shell_assignments())

    return 0


if __name__ == "__main__":
    sys.exit(main())
