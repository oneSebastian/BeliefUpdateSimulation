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
                    response parsing, permutation tests
scripts/
  pipeline/         run_agent.py (main entry point), academic_ai.py,
                    extract_initial_beliefs.py
  data/             excel_to_json.py, export_hf_dataset.py, and the
                    de-identification tools (pseudonymize, scan_text_pii,
                    strip_otree_identifiers)
  stats/            Permutation tests and statistics
  figures/          Figure generation

configs/           One JSON per model / temperature variant (22 total)
prompt_templates/  Prompt templates, ablation variants, topic map
data/              Inputs: personas and the cleaned survey exports
results/           Simulation outputs (.xlsx per model, plus ablations/)
outputs/           Generated stats, derived tables, and figures
slurm/             Cluster job scripts for locally served models
tests/             Test suite
colors.yaml        Shared per-model colour palette used by all figures
```

Everything is invoked as a module (`python -m scripts.…`). Input paths are
resolved from the repository root by `belief_update_sim.config` rather than from
the working directory. Figure scripts still write their images to a path
relative to the working directory, so **run them from the repository root**.

## Setup

```bash
pip install -r requirements.txt
pip install -e .        # makes belief_update_sim and scripts importable
cp .env.example .env    # then fill in keys for the providers you plan to use
```

Only the providers you actually run need keys. Locally served models (Qwen, Llama,
Olmo) need none — `scripts/pipeline/run_agent.py` talks to a vLLM
OpenAI-compatible endpoint on the `port` given in the config.

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

## License

Code is MIT — see `LICENSE`.

Data is Creative Commons Attribution 4.0 International (CC BY 4.0) — see
`LICENSE-DATA`. This covers `data/`, `results/`, the generated
`data/hf_dataset/` export, and the tables under `outputs/`. Attribution keeps
the participants' contribution traceable to the study they consented to, which a
public-domain dedication would not.
