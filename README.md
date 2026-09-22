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
| `--pilot` | Smoke test on 5 personas, written to a `pilot/` folder |
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

`slurm/run_model.slurm` reads that block, so one job script serves all twelve
models. Submit through the wrapper, which reads `--gpus`/`--time`/`--partition`
out of the same config (SLURM fixes those at submit time, not run time):

```bash
chmod +x slurm/submit_model.sh                          # once, on the cluster
./slurm/submit_model.sh --pilot-all                     # 5 personas per model
./slurm/submit_model.sh --all --resume                  # the full sweep
./slurm/submit_model.sh model_size/Qwen3.5-9B.json      # one model
```

**Batch modes are sequential by default.** Each job carries
`--dependency=afterany:<previous>`, so at most one model is resident at a time
and the rest of the node stays free for other users. `afterany` rather than
`afterok`: a model that OOMs or fails to load must not strand the eleven behind
it.

| Flag | Effect |
| --- | --- |
| *(default)* | chained — one model at a time |
| `--parallel` | no chain; SLURM runs as many as fit in the node's GPUs |
| `--after JOBID` | chain the first job behind an existing one |

Which lets you queue the full sweep behind the pilots — the last submitted job
id is printed for exactly this:

```bash
./slurm/submit_model.sh --pilot-all
#   ... Last job in the chain: 481203
./slurm/submit_model.sh --all --resume --after 481203
```

### Pilot walltime and the first-run download

Pilot jobs request the config's `serving.pilot_time`, not its full-run walltime
— 15 calls do not need 48 hours, and asking for them holds a slot the job cannot
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

`--pilot` runs 5 personas (15 calls) and writes to `results/model_size/pilot/`,
never touching the real output — so it stays available for configs whose output
has been disabled. It answers the two questions that are expensive to get wrong
on 391 personas:

```bash
python -m scripts.pipeline.pilot_report 'results/model_size/pilot/*.xlsx'
```

```
     model  n      valid  tokens med/max of budget tries  trunc  verdict
Qwen3.5-9B  9 9/9 (100%)   340/2180 of 16384 (13%)  1.00      0  OK
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

Validate a config without submitting anything:

```bash
python -m scripts.pipeline.serving_params --format human model_size/Qwen3.5-27B.json
python -m scripts.pipeline.serving_params --check configs/model_size/*.json
```

### GPU allocation

Sized for 4× H100 80GB on one node, BF16 weights at `gpu_memory_utilization`
0.90. `max_tokens` is 16K below 26B and 32K at and above it, with
`max_model_len` carrying another 4K for the ~2.2k-token prompt.

| Model | Weights (BF16) | GPUs | `max_tokens` |
| --- | --- | --- | --- |
| gemma-4-E2B-it | ~10 GB (5.1B raw / 2.3B effective) | 1 | 16384 |
| gemma-4-E4B-it | ~17 GB | 1 | 16384 |
| gemma-4-12B-it | ~24 GB | 1 | 16384 |
| gemma-4-26B-A4B-it | ~52 GB | 2 | 32768 |
| gemma-4-31B-it | ~62 GB | 2 | 32768 |
| Qwen3.5-0.8B / 2B / 4B / 9B | 2–18 GB | 1 | 16384 |
| Qwen3.5-27B | ~54 GB | 2 | 32768 |
| Qwen3.5-35B-A3B | ~70 GB | 2 | 32768 |
| Qwen3.5-122B-A10B | ~244 GB | 4 | 32768 |

MoE models keep all experts resident, so `26B-A4B` and `122B-A10B` are sized by
their *total* parameters, not their active ones.

**Qwen3.5-397B-A17B is not in the sweep.** It needs ~794 GB in BF16 and ~400 GB
in FP8; neither fits in 320 GB. Only `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (~200 GB)
would, which would make the largest point the only quantized one.

Qwen models are served with `--reasoning-parser qwen3`, which splits thinking
into `reasoning_content` and leaves clean JSON in `content`. Gemma has no such
parser configured, so its `<think>` blocks arrive inline and are stripped by
`_clean_json_text` — the path the existing Olmo runs already use.

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
