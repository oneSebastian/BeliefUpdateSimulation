# Human–LLM Agents Simulation

Code and data for the paper [**LLMs struggle to simulate human belief updates in controlled environments**](https://arxiv.org/abs/2607.28347) (arXiv:2607.28347).

This repository contains everything needed to simulate human survey participants with
LLM agents and to compare the simulated responses against the real human responses they
were derived from. 391 Prolific participants each gave an initial stance on three debate
topics (universal basic income, penalty shootouts, weight-loss drugs), read three
argumentative comments, ranked them, and gave a post-exposure stance. Each participant is
turned into a persona — demographics, Big-5 personality, initial stance — and replayed
through an LLM under the same conditions, so that human and simulated belief change can be
compared directly.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{pohl2026llmsstruggle,
  title         = {LLMs struggle to simulate human belief updates in controlled environments},
  author        = {Sebastian Pohl and Harsh Mehta and Pranav Mambayil and Abdul Ghafoor and Franziska Lesigang and Yufang Hou and Christian Hilbe},
  year          = {2026},
  eprint        = {2607.28347},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.28347},
  doi           = {10.48550/arXiv.2607.28347}
}
```

## Contents

```
belief_update_sim/  Importable core: config, data loading, normalization,
                    response parsing, permutation tests, cluster serving params
scripts/
  pipeline/         run_agent.py (main entry point), academic_ai.py,
                    extract_initial_beliefs.py, serving_params.py,
                    pilot_report.py
  data/             excel_to_json.py, export_hf_dataset.py, and the
                    de-identification tools (pseudonymize, scan_text_pii,
                    strip_otree_identifiers)
  stats/            Permutation tests and statistics
  figures/          Figure generation

configs/           One JSON per model / temperature variant (22 total), plus
                   model_size/ for the 12-model size sweep (34 total)
prompt_templates/  Prompt templates, ablation variants, topic map
data/              Inputs: personas and the cleaned survey exports
results/           Simulation outputs (.xlsx per model, plus ablations/ and
                   model_size/)
outputs/           Generated stats, derived tables, and figures
slurm/             Cluster job scripts for locally served models
tests/             Test suite
reference/         Static snapshot of the oTree survey application, included
                   for reference only — not imported or run by the pipeline
colors.yaml        Shared per-model colour palette used by all figures
```

Everything is invoked as a module (`python -m scripts.…`). Input paths are
resolved from the repository root by `belief_update_sim.config` rather than from
the working directory. Figure scripts still write their images to a path
relative to the working directory, so **run them from the repository root**.

## Requirements

**Operating systems (tested).** The analysis, statistics, and figure code was
developed and tested on **Windows 11**. Serving the open-weight models locally is
done on a **Linux** GPU cluster via SLURM; API-based models run on either.

**Python.** 3.10 or newer (tested on 3.14.2).

**Python packages.** Installed from `requirements.txt`, which is intentionally
left unpinned for portability. The versions the code has been tested with:

| Package | Tested version | | Package | Tested version |
| --- | --- | --- | --- | --- |
| pandas | 2.3.3 | | Pillow | 12.2.0 |
| numpy | 2.4.1 | | svgutils | 0.3.4 |
| scipy | 1.17.0 | | PyMuPDF | 1.28.0 |
| statsmodels | 0.14.6 | | openai | 2.46.0 |
| matplotlib | 3.10.9 | | anthropic | 0.117.1 |
| seaborn | 0.13.2 | | google-genai | 2.13.0 |
| openpyxl | 3.1.5 | | requests | 2.32.5 |
| pyarrow | 23.0.0 | | python-dotenv | 1.2.2 |
| PyYAML | 6.0.3 | | | |

**Hardware.** No special hardware is required for the analysis, statistics, and
figures, or for API-based simulation — any normal desktop CPU is sufficient, and
there are no particular CPU requirements. Serving the open-weight models locally
(Llama-3.3-70B, Qwen3-32B, and the twelve models of the size sweep) is the only
step that needs a GPU: our runs used **4× NVIDIA H100 80GB HBM3** GPUs with vLLM
tensor parallelism (see `slurm/run_Llama70b.slurm`: `--gpus=4`,
`--tensor-parallel-size 4`). The size sweep sizes its allocation per model from
each config's `serving` block — see [Model-size sweep](#model-size-sweep).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda; needs Python >= 3.10
pip install -r requirements.txt
pip install -e .        # makes belief_update_sim and scripts importable
cp .env.example .env    # then fill in keys for the providers you plan to use
```

Installation takes a few minutes on a normal desktop (typically 2–5 minutes,
dominated by the scientific-stack wheels).

Only the providers you actually run need keys. Locally served models (Qwen, Llama,
Olmo) need none — `scripts/pipeline/run_agent.py` talks to a vLLM
OpenAI-compatible endpoint on the `port` given in the config.

## Demo

A self-contained example that needs no API keys and no GPU — it runs on the
survey data shipped in this repository. From the repository root:

```bash
python -m scripts.figures.distributions.post_stance_models
```

**Expected output.** A six-panel figure (one panel per model) of the
post-exposure stance distribution, each overlaid with the human post-stance
baseline, written to
`figures/plots/distributions/post_stance_models.{png,pdf,eps,svg}` (plus the
individual `panel_*` panels in the same folder).

**Expected run time.** About 10 seconds on a normal desktop.

## 1. Simulate

```bash
python -m scripts.pipeline.run_agent --config gpt-5.2.json --resume
```

`--resume` skips personas already present in the output workbook, so an interrupted
run can be restarted safely. Results go to the `output_excel` path declared in the
config; per-persona progress is appended to `results/progress_log.jsonl`.

| Flag | Effect |
| --- | --- |
| `--config NAME` | Config file in `configs/` (default `gpt-5.2.json`) |
| `--resume` | Skip personas already in the output workbook |
| `--single_persona ID` | Run one persona folder only |
| `--single_topic T` | Run one topic only (`UBI`, `penalty`, `weight_loss`) |
| `--ablation NAME` | Run an ablation (see below) |
| `--limit N` | Run only the first N personas (the list is sorted, so N is stable) |
| `--pilot` | Smoke test on 3 personas, written to a `pilot/` folder |
| `--print_prompt` | Print the assembled prompt |

With `--ablation`, output is written to `results/ablations/{model}_{NAME}.xlsx`
instead of the config's path. Supported values:

`all-positive`, `all-negative`, `no-demographic`, `no-personality`, `no-persona`,
`probe-initial-belief`, `own-initial-belief`

Temperature ablations use dedicated configs instead (`*-t0.0.json`, `*-t2.0.json`).

For locally served models, submit the matching SLURM script — it starts a vLLM
server, waits for it to become healthy, runs the agent, then shuts the server down:

```bash
sbatch slurm/run_qwen32.slurm
```

## Model-size sweep

`configs/model_size/` holds twelve locally served models — Gemma 4 (E2B, E4B,
12B, 26B-A4B, 31B) and Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B) —
run under identical conditions so that parameter count is the only thing that
varies. Results go to `results/model_size/`.

**Held constant across the sweep**, and checked by `tests/test_serving.py`:
temperature 0.7, `templates2.txt`, the full 391-persona set, `max_tries` 3, and
`enable_thinking: true`. The two families default in opposite directions —
Qwen3.5 thinks unless told not to, Gemma 4 only when asked — so reasoning mode
is forced on for every model rather than left to the chat template, which would
otherwise confound model family with reasoning mode.

Each config carries a `serving` block with everything the cluster needs:

```json
"serving": {
  "hf_model_id": "Qwen/Qwen3.5-9B",
  "gpus": 1, "tensor_parallel_size": 1,
  "max_model_len": 20480, "gpu_memory_utilization": 0.90,
  "partition": "verylong", "time": "12:00:00",
  "reasoning_parser": "qwen3", "extra_args": []
}
```

An optional `"mem"` sets the job's **host** RAM (`--mem`), which is unrelated to
`gpu_memory_utilization`. Omit it to take the partition default, as eleven of
the twelve configs do. Only `Qwen3.5-122B-A10B` declares one (`"500G"`), because
vLLM prefetches the whole checkpoint into the page cache before loading it and
the page cache is charged to the job's cgroup — with 233 GiB of weights across
four ranks, the default was not enough and the job was OOM-killed *after* a
26-minute download, reported only as `Worker proc VllmWorker-3 died
unexpectedly` plus a `slurmstepd: Detected 1 oom-kill event(s)` at the very end
of the job's stderr.

`slurm/run_model.slurm` reads that block, so one job script serves all twelve
models. Submit through the wrapper, which reads
`--gpus`/`--time`/`--partition`/`--mem` out of the same config (SLURM fixes
those at submit time, not run time):

```bash
chmod +x slurm/submit_model.sh                          # once, on the cluster
./slurm/submit_model.sh --pilot-all                     # 3 personas per model
./slurm/submit_model.sh --all --resume                  # the full sweep
./slurm/submit_model.sh model_size/Qwen3.5-9B.json      # one model
```

**Batch modes are packed into waves.** Jobs are greedily grouped so each wave
totals at most `--max-gpus` (default 4, the size of the node). A wave's jobs run
concurrently; every job in wave N+1 carries
`--dependency=afterany:<all of wave N>`:

```
  -- wave 1 --   Qwen3.5-0.8B  Qwen3.5-2B  Qwen3.5-4B  gemma-4-E2B-it     4 GPUs
  -- wave 2 --   gemma-4-E4B-it  Qwen3.5-9B  gemma-4-12B-it               3 GPUs
  -- wave 3 --   Qwen3.5-27B  Qwen3.5-35B-A3B                             4 GPUs
  -- wave 4 --   gemma-4-26B-A4B-it  gemma-4-31B-it                       4 GPUs
  -- wave 5 --   Qwen3.5-122B-A10B                                        4 GPUs
```

The dependency is what makes this neighbourly. Only wave 1 ever requests
resources; the rest sit in `PENDING` with reason `Dependency`, which SLURM does
not schedule or backfill against — so they cannot hold GPUs away from other
users the way a queue of resource-pending jobs would. Check it with:

```bash
squeue -u $USER -o '%.10i %.24j %.8T %.20r %.6b'
```

Each job depends on the *whole* previous wave, not just its last job — waiting
on one would let wave 2 start while wave 1 still had jobs running, over-
subscribing the node. `afterany` rather than `afterok`: a model that OOMs or
fails to load must not strand the waves behind it.

**Waves are filled cheapest-allocation-first**, so the 4-GPU model lands last
and the eleven ahead of it have produced results before it waits on a full node.
Within a GPU tier the order is by weight-download size; ties break on name, so a
resubmission reproduces the same plan. Greedy packing leaves wave 2 one GPU
short — a tighter bin-pack still needs five waves, so it buys nothing but
unpredictability.

A job that needs more GPUs than `--max-gpus` takes a wave to itself rather than
being dropped: `--max-gpus 2` cannot shrink the 4-GPU model.

| Flag | Effect |
| --- | --- |
| *(default)* | waves of at most 4 GPUs |
| `--max-gpus N` | change the wave capacity |
| `--sequential` | one job at a time; the same as `--max-gpus 1` |
| `--parallel` | no dependencies at all — the one mode where queued jobs *do* request resources |
| `--after JOBID` | hold the first wave behind an existing job |
| `--only GLOB` | restrict the batch, e.g. `--only 'gemma-4-*'` to re-run one family |

Which lets you queue the full sweep behind the pilots — the last submitted job
id is printed for exactly this:

```bash
./slurm/submit_model.sh --pilot-all
#   ... Last job in the chain: 481203
./slurm/submit_model.sh --all --resume --after 481203
```

### Pilot walltime and the first-run download

Pilot jobs request the config's `serving.pilot_time`, not its full-run walltime
— 9 calls do not need 48 hours, and asking for them holds a slot the job cannot
use and loses backfill priority.

`pilot_time` is sized **per model**, because the first run of each downloads its
weights — 2 GB for Qwen3.5-0.8B against 244 GB for 122B-A10B, roughly 565 GB
across the sweep. A single shared value would either strand the large download
or make the small models over-reserve. The values assume a pessimistic ~10 MB/s,
so a cold 244 GB fetch still fits:

| Weights | Models | `pilot_time` |
| --- | --- | --- |
| ≤ 10 GB | Qwen3.5-0.8B / 2B / 4B | 02:00:00 |
| 10–20 GB | Qwen3.5-9B, gemma-4-E2B/E4B | 03:00:00 |
| 20–40 GB | gemma-4-12B | 04:00:00 |
| 40–80 GB | Qwen3.5-27B / 35B-A3B, gemma-4-26B-A4B / 31B | 06:00:00 |
| ~244 GB | Qwen3.5-122B-A10B | 12:00:00 |

Once weights are cached these are very generous — but a job ends when it ends,
so only the *reservation* is bounded. On a slower link, override every model at
once, or pre-fetch and keep the reservations small:

```bash
PILOT_TIME_LIMIT=24:00:00 ./slurm/submit_model.sh --pilot-all
hf download Qwen/Qwen3.5-122B-A10B      # the alternative
```

`tests/test_serving.py` checks that every sweep config declares `pilot_time`,
that it is monotone in model size, and that it stays below the full-run
walltime.

### Pilot first

`--pilot` runs 3 personas x 3 topics = 9 calls, writing to `results/model_size/pilot/`,
never touching the real output — so it stays available for configs whose output
has been disabled. It answers the two questions that are expensive to get wrong
on 391 personas:

```bash
python -m scripts.pipeline.pilot_report 'results/model_size/pilot/*.xlsx'
```

```
     model  n      valid  tokens med/max of budget tries  trunc  verdict
Qwen3.5-9B  9 9/9 (100%)   340/2180 of 32768 (7%)  1.00      0  OK
```

A model can fail to produce usable output for two reasons that look identical in
the parsed columns, so the report separates them by stop reason:

| Signal | Meaning | Fix |
| --- | --- | --- |
| `finish_reason == "length"` | ran out of completion budget mid-thought | raise `max_tokens` (and `max_model_len` with it) |
| unparseable, stopped normally | ignored the JSON format instructions | prompt or model capability — for the smallest models this is a *result*, not a bug |

Truncation is counted over **attempts**, not final rows: a retry that recovers
still means the budget was too small, it just paid for the discovery twice. The
report also flags responses that used more than 90% of the budget without
truncating, since those will truncate on the full run. It exits non-zero when
any model needs attention, so a pilot job fails visibly rather than printing a
warning into a log nobody reads.

The `no-think` column counts rows answered by the fallback described below; a
model with rows there gets a `NO-THINKING FALLBACK` verdict rather than
`RAISE max_tokens`, because for those rows the ceiling was not the problem.

### The no-thinking fallback

When two consecutive attempts on the same persona × topic both end with
`finish_reason == "length"`, the remaining attempts for that row are made with
`enable_thinking: false`, and the row records it:

| Column | Meaning |
| --- | --- |
| `thinking_disabled` | the answer stored in this row was produced with thinking off |
| `n_no_thinking` | how many of the row's attempts ran that way |

The pilots are what motivates this. Across Qwen3.5-0.8B/2B, every response that
ever parsed finished under ~12k tokens, while the ones that truncated ran away
past 19k to the 32k ceiling — there is no mass in between, so a third *thinking*
attempt on an already-truncated row buys another full budget of runaway rather
than an answer. (Doubling 0.8B's ceiling from 16K to 32K changed nothing: 0/3
valid both times, every attempt at the ceiling.)

Two details matter. The switch only applies to configs that asked for thinking
in the first place — a plain API model has no chat template to flip. And it
**latches** for the rest of the row: a no-thinking attempt ends with
`finish_reason == "stop"` by design, so treating that as "the runaway broke"
would hand the next attempt back to thinking and straight back into it.

`thinking_disabled` exists because these rows are not comparable to the rest —
they answer a different question than the one the sweep asks. Filter or flag
them in analysis rather than pooling them.

Validate a config without submitting anything:

```bash
python -m scripts.pipeline.serving_params --format human model_size/Qwen3.5-27B.json
python -m scripts.pipeline.serving_params --check configs/model_size/*.json
```

### GPU allocation

Sized for 4× H100 80GB on one node, BF16 weights at `gpu_memory_utilization`
0.90.

| Model | Weights (BF16) | GPUs |
| --- | --- | --- |
| gemma-4-E2B-it | ~10 GB (5.1B raw / 2.3B effective) | 1 |
| gemma-4-E4B-it | ~17 GB | 1 |
| gemma-4-12B-it | ~24 GB | 1 |
| gemma-4-26B-A4B-it | ~52 GB | 2 |
| gemma-4-31B-it | ~62 GB | 2 |
| Qwen3.5-0.8B / 2B / 4B / 9B | 2–18 GB | 1 |
| Qwen3.5-27B | ~54 GB | 2 |
| Qwen3.5-35B-A3B | ~70 GB | 2 |
| Qwen3.5-122B-A10B | ~244 GB | 4 |

### Token budget

`max_tokens` is **16384 for every model**, with `max_model_len` at 36864 —
comfortably more than the 4K the ~2.2k-token prompt needs. It is deliberately
uniform: an earlier tiering (16K below 26B, 32K at and above) gave the *small*
models a smaller budget than the large ones, confounding budget with size in a
sweep whose whole premise is that only size varies.

The level was briefly 32768, on the reasoning that a larger ceiling costs a
healthy model nothing because a model that answers stops when it is done. That
is true per *call* and false per *run*. The ceiling is precisely what a runaway
spends, and a run's cost is dominated by runaways: at 32768, Qwen3.5-0.8B spent
two full 32k attempts on 8 of 9 pilot rows before a third attempt answered in
~150 tokens, which is ~52 h of walltime against ~26 h at 16384. Meanwhile no
valid response in any of the seven pilots exceeded **13,257** tokens:

| model | largest *valid* response |
| --- | --- |
| Qwen3.5-0.8B | 13,257 |
| Qwen3.5-2B | 11,539 |
| Qwen3.5-4B | 8,763 |
| Qwen3.5-9B | 6,888 |
| gemma-4-12B-it | 4,598 |

So the headroom above 16K was never reached by anything that worked, and was
paid for twice per failing row. Raise it only against evidence of a *valid*
response near the ceiling — which is what pilot_report's
`tokens med/max of budget` column exists to show — not against truncations,
which are a capability limit at these sizes rather than a budget one.

### Full-run walltime

The seven single-GPU models have a measured pilot, so their `serving.time` is
set from it rather than guessed. Two independent estimates were used and agreed
within ~5%: measured agent walltime per row scaled to the full 1173 rows, and
generated tokens per row over the decode rate vLLM reported. Estimates are at
`max_tokens` 16384 and include server startup.

| model | estimated | requested |
| --- | --- | --- |
| gemma-4-E2B-it | ~3 h | 06:00:00 |
| gemma-4-E4B-it | ~4 h | 08:00:00 |
| Qwen3.5-4B | ~10 h | 16:00:00 |
| Qwen3.5-9B | ~12 h | 18:00:00 |
| gemma-4-12B-it | ~12 h | 20:00:00 |
| Qwen3.5-2B | ~20 h | 32:00:00 |
| Qwen3.5-0.8B | ~26 h | 40:00:00 |

The margin is wide on purpose: each estimate rests on 9 pilot rows from the
first three personas, and for the two slow models the runtime is driven almost
entirely by the truncation rate, whose 95% interval is 57–98% for Qwen3.5-0.8B.
The five multi-GPU models keep their original guesses, having never completed a
pilot.

Note that 9B and 12B were at ~12 h against a 12:00:00 request — their pilots
passed with 100% validity and zero truncations, so nothing flagged them, and
they would have been killed near the end of the full run. A clean pilot says
nothing about walltime.

MoE models keep all experts resident, so `26B-A4B` and `122B-A10B` are sized by
their *total* parameters, not their active ones.

**Qwen3.5-397B-A17B is not in the sweep.** It needs ~794 GB in BF16 and ~400 GB
in FP8; neither fits in 320 GB. Only `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (~200 GB)
would, which would make the largest point the only quantized one.

Each family is served with its own reasoning parser — `qwen3` and `gemma4` —
which splits thinking into `reasoning_content` and leaves clean JSON in
`content`. If a vLLM build lacks one, drop `reasoning_parser` from the config:
the `<think>` block then arrives inline and is stripped by `_clean_json_text`,
the path the existing Olmo runs already use.

### Serving requirements

**Gemma 4 needs vLLM ≥ 0.27.2rc0.** Earlier versions read `head_dim` off the
config globally, which transformers ≥ 5.15 refuses because Gemma 4's genuinely
varies per layer. On 0.27.1 every Gemma job dies at engine start with:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
```

```bash
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129
```

Do **not** take the workaround the error message suggests
(`allow_global_per_layer_attribute_access=True`). It silences the exception by
handing vLLM a single `head_dim` for a model whose layers genuinely differ, so
the server starts and then computes against the wrong geometry — a wrong number
instead of a crash. Pinning `transformers==5.14.1` also clears it, but pins you
to an older transformers to serve a model that needs a newer one.

**FlashInfer kernels are disabled, because the cluster's nvcc is too old.**
FlashInfer JIT-compiles kernels at runtime using whatever `nvcc` is on `PATH`.
The system toolkit cannot build them, and the build fails *after* the weights
have loaded — ten minutes into a job that then dies. Two forms of the failure
have been seen:

```
nvcc fatal : Unsupported gpu architecture 'compute_90a'
nvcc fatal : Unknown option '--compress-mode=size'
```

Three JITs are affected:

| JIT | Disabled by | Where |
| --- | --- | --- |
| top-k/top-p sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` | `slurm/run_model.slurm` |
| fused all-reduce + RMSNorm (tensor-parallel only) | `VLLM_ALLREDUCE_USE_FLASHINFER=0` | `slurm/run_model.slurm` |
| gated-delta-net prefill (Qwen3.5 only) | `--gdn-prefill-backend triton` | the Qwen configs' `extra_args` |

The first two live in the job script rather than per config, so every model in
the sweep runs the same kernels — a kernel that varied across models would be a
confound on the size axis. The fallback paths draw from the same distributions,
so results are unaffected; they are somewhat slower, which does not matter for
1173 short calls, and it removes runtime compilation from the batch jobs
entirely.

The all-reduce one is worth understanding before changing it, because it looks
like a tensor-parallelism problem and is not. vLLM 0.30 promoted `FLASHINFER`
to the front of the all-reduce dispatch order; 0.27.1 kept it in the candidate
list and chose `CUSTOM`. The fused all-reduce+RMSNorm pass only builds
`trtllm_mnnvl_allreduce.cu` once dispatch actually goes through FlashInfer — a
working 0.27.1 run on 4 GPUs had `fuse_allreduce_rms: True` and never invoked
`nvcc` at all. So do **not** "fix" this by disabling the fusion pass
(`-O.pass_config.fuse_allreduce_rms=false`); that removes an optimization the
older runs had. It is the backend choice that regressed, and
`VLLM_ALLREDUCE_USE_FLASHINFER=0` is what restores the old behaviour.

To undo all three once a CUDA ≥ 12.8 toolkit is available
(`conda install -c nvidia cuda-nvcc`): set `VLLM_USE_FLASHINFER_SAMPLER=1` and
`VLLM_ALLREDUCE_USE_FLASHINFER=1`, and drop `--gdn-prefill-backend` from
`extra_args`.

Re-run just the affected family after a fix, rather than the whole sweep:

```bash
./slurm/submit_model.sh --pilot-all --only 'gemma-4-*'
```

**On your own data.** To simulate a different participant set, place each persona
under `data/prolific_data/<id>/` as `demographic.json` + `study_data.json` (the
layout documented under [Data](#data)); the commands above then apply unchanged,
and `--single_persona ID` runs just one.

## 2. Analyze

Permutation tests comparing human against simulated belief change:

```bash
python -m scripts.stats.ablations
python -m scripts.stats.analyze
```

Writes `.parquet` and `.json` tables to `outputs/stats/`. Which result file feeds
which comparison is declared in `scripts/stats/data_paths.json`.

`belief_update_sim.data_loading` is the shared loader and the reason human and LLM
data are comparable at all. It sign-flips negatively phrased topic variants so that
positive always means "toward the proposition", and renames columns onto a common
schema (`init_stance`, `new_belief`, `general_public_stance`). Both human and LLM
data pass through it.

## 3. Figures

Run any figure script as a module:

```bash
python -m scripts.figures.consistency_across_llms
python -m scripts.figures.distributions.human_distributions
python -m scripts.figures.stances
```

Output lands in `figures/plots/` as `.png` and `.pdf`, often also `.svg`/`.eps`.
Model colours come from `colors.yaml`, keeping panels consistent across figures.

Two scripts build intermediate tables that other scripts consume, so run them first:

```bash
python -m scripts.figures.generate_zscore_csv
python -m scripts.figures.zscore_initial_beliefs_combined
```

Everything in `outputs/derived/` is a regenerable intermediate, not a source input.

## Reproducing the reported statistics

The statistics reported in the paper regenerate from the shipped data with no API
keys or GPU. From the repository root:

```bash
python -m scripts.stats.post_stance_distribution   # chi-squared: human vs LLM post-stance
python -m scripts.stats.belief_change_variability  # Brown-Forsythe on |belief change|
python -m scripts.stats.comment_rank_correlation   # Kendall tau on comment rankings
python -m scripts.stats.comment_rank_variability   # Brown-Forsythe on comment mean-ranks
```

Each writes a `.txt` and `.json` summary to `outputs/stats/`.

## Data

| Path | Contents |
| --- | --- |
| `data/prolific_data/` | 391 personas — `demographic.json` + `study_data.json` each |
| `data/cleaned_for_llm_391_participant_otree.csv` | Cleaned oTree survey export, used by the pipeline |
| `data/cleaned_391_participant_otree.xlsx` | The same export as delivered by oTree |
| `data/400_participant_otree.xlsx` | Full oTree export, before exclusions |
| `data/hf_dataset/` | Generated HuggingFace export (gitignored; rebuild with `python -m scripts.data.export_hf_dataset`) |

### Comment-ranking conventions

Participants and models both ranked the three comments they were shown, but the
two sources record it differently, and **the columns look identical** — each is
a permutation of {1, 2, 3}. Read the wrong one and you get plausible but wrong
results rather than an error, so check this table before touching a `rank_*`
column.

| File | Column | The value is | Indexed by |
| --- | --- | --- | --- |
| `results/*.xlsx` (models) | `rank_1/2/3` | **the comment id** placed i-th | rank position |
| `results/merged_..._normalized.csv` (humans) | `rank_message_N_shown` | **the rank** given to that comment | shown slot |
| `data/*_otree.xlsx` (raw survey) | `rank_topic_T_m` | **the rank** given to that comment | source message |
| `data/hf_dataset/` (both) | `rank_1/2/3` | **the rank** given to that comment | shown slot |

`1` is always the most persuasive.

Models were shown the comments as `Comment 1/2/3` in a per-participant random
order and asked to list those ids most- to least-convincing, so a model's
`rank_1` holds a *comment id*, not a rank. Humans assigned a position to each
comment in a grid, so their columns hold *ranks*. The two are inverse
permutations of one another, and four of the six permutations of three items are
their own inverse — so the two readings agree on two thirds of rows.

`belief_update_sim.ranking.ranks_from_model_ordering()` converts model rows into
the human convention and is the only place that conversion is implemented. Use
it whenever model and human rankings are compared; leave rankings alone when
only round-tripping a model's own output. The exported dataset applies it, so
`data/hf_dataset/` is uniform.

Ranks are recorded per **shown slot**. To aggregate per comment, map the slot to
its source message — `shown_messages` in the dataset, `message_N_shown` in the
merged CSV, `message_order_perm` in the results workbooks.

### Participant privacy

Participants are identified only by pseudonyms of the form `P0001`–`P0429`,
assigned by a seeded shuffle so the numbering carries no information about
recruitment order or any participant attribute. The recruitment-platform
identifiers they replaced were removed from every file in this repository, and
the key linking the two is held privately and is not published. Platform
session tokens and per-participant timestamps were dropped from the oTree
exports for the same reason.

Records contain self-reported demographics, Big-5 responses, stances, rankings,
and free-text answers. Demographics are retained because they are the
experimental conditioning variable, and the free text has been screened for
self-identifying content. This is pseudonymisation, not anonymisation:
demographics are quasi-identifiers, and the free text is participants' own
writing.

## Ethics and consent

Prior to data collection this project has been reviewed by the research ethics
committee of the Interdisciplinary Transformation University Austria under the
case number 2025-09.

Participants were recruited through Prolific and gave informed consent on a
form shown before any data was collected. The form stated the purpose of the
study (investigating the behaviour of humans and LLM agents), that participants
would not interact with an LLM at any point, that demographic information was
collected in order to condition LLM agents on a comparable distribution in a
later study, the approximate duration, and that participation was paid.

They were informed that pseudonymised data would be released publicly and would
not permit their identification, and that they could discontinue the study at
any time without penalty. Participants who did not consent were routed out of
the study before any responses were recorded, and returned submissions are
excluded from the released data (391 of 400 recruited participants remain).

The consent form and the survey instrument are the oTree application included in
this repository under `reference/otree_survey/` (for reference only).

## License

Code is MIT — see `LICENSE`.

Data is Creative Commons Attribution 4.0 International (CC BY 4.0) — see
`LICENSE-DATA`. This covers `data/`, `results/`, the generated
`data/hf_dataset/` export, and the tables under `outputs/`. Attribution keeps
the participants' contribution traceable to the study they consented to, which a
public-domain dedication would not.
