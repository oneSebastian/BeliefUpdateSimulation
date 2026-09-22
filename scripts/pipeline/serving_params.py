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
    ServingConfigError,
    load_serving_config,
    parse_slurm_time,
)


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
    args = parser.parse_args(argv)

    if len(args.config) > 1 and not (args.check or args.sort_by_gpus):
        parser.error("multiple configs are only supported with --check/--sort-by-gpus")

    if args.sort_by_gpus:
        try:
            servings = [(c, load_serving_config(c)) for c in args.config]
        except ServingConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for config, _ in sorted(servings, key=lambda pair: sort_key(pair[1])):
            print(config)
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
