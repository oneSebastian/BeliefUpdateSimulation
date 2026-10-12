"""Rank the LLMs by how faithfully they simulate the humans.

The permutation test (belief_update_sim.permutation_test) answers a yes/no
question -- does a model's mean post-stance differ from the humans'? -- and says
nothing about which model is closer. Here every model is scored on the same
persona x topic cases, with uncertainty on both the scores and the ranks.

1. Error metrics, lower is better, on the normalised 5-point scale
       mae          mean |human post-stance - model post-stance|. The primary
                    metric: how well the model predicts *this* person
       wasserstein  earth mover's distance between the two post-stance
                    distributions. Population level; ignores who is who
       spread       |log(SD_model / SD_human)| of the absolute belief change
                    |post - initial|, the quantity the Brown-Forsythe test compares
   plus bias = mean(human - model), the quantity the permutation test tests,
   reported with a CI but not ranked (it is signed).

2. Participant-level cluster bootstrap. Participants are resampled with
   replacement, carrying all their topics along, and every source is scored on
   the same resample, so the comparison stays paired. Gives a percentile CI per
   score, a CI for each model's rank among the LLMs, and P(best).

3. Omnibus test on MAE: do the LLMs differ at all? Under the null every
   model is equally close to the humans, so within a participant the model
   labels on the absolute errors are exchangeable. The statistic is the
   variance of the models' MAEs; the null distribution comes from permuting
   the labels independently per participant (one permutation shared by all of
   that participant's topics, which are correlated). Only the LLMs enter it.

4. Pairwise tests on MAE. For each pair of sources the per-case difference in
   absolute error is summed per participant and tested with a two-sided
   sign-flip permutation test that flips whole participants. The raw p is
   reported; a pair counts as significant below the Bonferroni threshold
   alpha / number of pairs. The result is summarised as a compact letter
   display: sources sharing a letter are not significantly different.

5. Baselines, optional (``--baselines``), scored like models but not ranked
       No change          predicts the initial belief
       Cell median (LOO)  median post-stance of the *other* humans in the same
                          topic x package x initial-belief cell -- "someone like
                          you". A model that does not beat it adds nothing about
                          the individual beyond the cell.
   With baselines the pairwise tests include them, so there are more pairs and
   the Bonferroni threshold is stricter.

A model starts from the human's initial belief by design, so belief change is
measured against the human's normalised initial belief for every source. Cases
where any model's response did not parse are dropped for all sources, so every
score is computed on identical cases.

Writes model_ranking.txt, .json and .tex (the omnibus and pairwise results as
a LaTeX table).

    python -m scripts.stats.model_ranking
    python -m scripts.stats.model_ranking --baselines
    python -m scripts.stats.model_ranking --n-boot 2000 --n-perm 20000
"""

import argparse
import itertools
import json
import string

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data

from scripts.stats.permutation_by_topic import MODEL_LABELS, num, p_cell

NO_CHANGE = "No change"
CELL_MEDIAN = "Cell median (LOO)"
BASELINES = (NO_CHANGE, CELL_MEDIAN)
CELL = ["topic", "package", "init"]

METRICS = {
    "mae": "mean |human - model| post-stance (individual level)",
    "wasserstein": "earth mover's distance between post-stance distributions (population level)",
    "spread": "|log(SD_model / SD_human)| of |belief change| (spread)",
}
PRIMARY = "mae"

ALPHA = 0.05            # family-wise, Bonferroni across all pairs
N_BOOT = 10_000
N_PERM = 100_000
SEED = 20260928
BOOT_CHUNK = 500        # bootstrap replicates per matrix product
PERM_CHUNK = 10_000     # sign-flip draws per matrix product

RULE = "=" * 104
LINE = "-" * 104


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def loo_cell_median(frame):
    """Median post-stance of the *other* humans in each case's CELL."""
    out = pd.Series(np.nan, index=frame.index)
    for key, g in frame.groupby(CELL):
        if len(g) < 2:
            raise ValueError(f"cell {key} holds a single case; "
                             "no leave-one-out median is defined")
        values = g["human"].to_numpy(dtype=float)
        for i, idx in enumerate(g.index):
            out[idx] = np.median(np.delete(values, i))
    return out


def assemble(human, models):
    """One row per persona x topic: the human, every model and the baselines.

    human   -- persona_id, topic, package, init_stance, new_belief (normalised),
               as load_human_normalized_data returns
    models  -- {name: frame with persona_id, topic, package, new_belief
               (normalised)}, as load_normalized_data returns

    Returns (frame, dropped): frame has columns persona_id, topic, package,
    init, human, one per model and one per baseline; dropped counts, per model,
    the cases removed because its response did not parse.
    """
    frame = (human[["persona_id", "topic", "package", "init_stance", "new_belief"]]
             .rename(columns={"init_stance": "init", "new_belief": "human"})
             .reset_index(drop=True))
    if frame[["init", "human"]].isna().any().any():
        raise ValueError("human data has missing initial or final beliefs")

    for name, df in models.items():
        m = frame[["persona_id", "topic", "package"]].merge(
            df[["persona_id", "topic", "package", "new_belief"]],
            on=["persona_id", "topic"], how="left", suffixes=("", "_m"),
            validate="one_to_one", indicator=True)
        missing = int((m["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"{name}: {missing} persona-topic cases are missing "
                             "from the model results")
        mismatched = int((m["package"] != m["package_m"]).sum())
        if mismatched:
            raise ValueError(f"{name}: {mismatched} persona-topic cases were shown "
                             "a different comment package than the human")
        frame[name] = pd.to_numeric(m["new_belief"], errors="coerce").to_numpy(dtype=float)

    frame[NO_CHANGE] = frame["init"].astype(float)
    frame[CELL_MEDIAN] = loo_cell_median(frame)

    dropped = {name: int(frame[name].isna().sum()) for name in models}
    frame = frame.dropna(subset=list(models)).reset_index(drop=True)
    return frame, dropped


def load():
    models = {}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        models[name] = load_normalized_data(path)
    return assemble(load_human_normalized_data(), models)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _arrays(frame, sources):
    h = frame["human"].to_numpy(dtype=float)
    p = frame[list(sources)].to_numpy(dtype=float)
    init = frame["init"].to_numpy(dtype=float)
    return h, p, init


def participant_codes(frame):
    return pd.factorize(frame["persona_id"])[0]


def score(weights, h, p, init):
    """Every metric for every source, under each row of case weights.

    weights -- (B, n) non-negative case weights. A row of ones is the plain
               sample; a bootstrap replicate weights each case by how often
               its participant was drawn.
    h, init -- (n,) human post-stance and initial belief
    p       -- (n, S) post-stance of each source
    Returns {metric: (B, S)} for every METRICS key plus "bias".
    """
    w = np.asarray(weights, dtype=float)
    sw = w.sum(axis=1, keepdims=True)
    err = h[:, None] - p
    out = {"mae": (w @ np.abs(err)) / sw, "bias": (w @ err) / sw}

    # W1 = sum over the support of |F_human - F_model| times the gap to the
    # next support point; the CDFs are weighted means of indicators
    grid = np.unique(np.concatenate([h, p.ravel()]))
    at, gaps = grid[:-1], np.diff(grid)
    f_h = (w @ (h[:, None] <= at).astype(float)) / sw                     # (B, K)
    ind_p = (p[:, :, None] <= at).reshape(len(h), -1).astype(float)      # (n, S*K)
    f_p = ((w @ ind_p) / sw).reshape(len(w), p.shape[1], len(at))       # (B, S, K)
    out["wasserstein"] = (np.abs(f_p - f_h[:, None, :]) * gaps).sum(axis=2)

    def sd(a):
        mean = (w @ a) / sw
        return np.sqrt(np.clip((w @ a ** 2) / sw - mean ** 2, 0, None))

    sd_h = sd(np.abs(h - init)[:, None])
    sd_p = sd(np.abs(p - init[:, None]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["spread"] = np.abs(np.log(sd_p / sd_h))
    return out


def bootstrap(frame, sources, n_boot=N_BOOT, seed=SEED):
    """{metric: (n_boot, S)} over participant-level bootstrap replicates."""
    h, p, init = _arrays(frame, sources)
    codes = participant_codes(frame)
    n_part = codes.max() + 1
    rng = np.random.default_rng(seed)
    reps = {}
    for start in range(0, n_boot, BOOT_CHUNK):
        size = min(BOOT_CHUNK, n_boot - start)
        draws = rng.multinomial(n_part, np.full(n_part, 1 / n_part), size=size)
        for k, v in score(draws[:, codes], h, p, init).items():
            reps.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in reps.items()}


def _ci(x, axis=0):
    # inverted_cdf picks observed values, so an all-inf column stays inf
    # instead of interpolating to nan
    return np.quantile(x, [0.025, 0.975], method="inverted_cdf", axis=axis)


def summarize(point, reps, sources, llms):
    """{metric: {source: row}}: estimate and 95% CI for every source; for the
    ranked metrics each LLM also gets its rank among the LLMs, a rank CI and
    P(best). Ties share the better rank."""
    llm_idx = [sources.index(n) for n in llms]
    table = {}
    for metric in (*METRICS, "bias"):
        lo, hi = _ci(reps[metric])
        ranked = metric in METRICS
        if ranked:
            rank_reps = rankdata(np.nan_to_num(reps[metric][:, llm_idx], nan=np.inf),
                                 axis=1, method="min")
            point_rank = rankdata(np.nan_to_num(point[metric][llm_idx], nan=np.inf),
                                  method="min")
        rows = {}
        for i, name in enumerate(sources):
            row = {"estimate": float(point[metric][i]), "ci": [float(lo[i]), float(hi[i])]}
            if ranked and name in llms:
                j = llms.index(name)
                r = rank_reps[:, j]
                r_lo, r_hi = _ci(r)
                row.update(rank=int(point_rank[j]), rank_ci=[int(r_lo), int(r_hi)],
                           p_best=float((r == 1).mean()))
            rows[name] = row
        table[metric] = rows
    return table


# --------------------------------------------------------------------------
# pairwise tests
# --------------------------------------------------------------------------

def cluster_sign_flip(diffs, codes, n_perm=N_PERM, seed=SEED):
    """Two-sided sign-flip test of mean(diffs) == 0, flipping whole participants.

    diffs -- (n, m) per-case differences, one column per comparison
    codes -- (n,) participant index of each case
    Returns (observed mean per column, p-value per column). p is
    (exceedances + 1) / (n_perm + 1), so it is never exactly 0.
    """
    diffs = np.asarray(diffs, dtype=float)
    n_part = codes.max() + 1
    totals = np.zeros((n_part, diffs.shape[1]))
    np.add.at(totals, codes, diffs)
    obs = totals.sum(axis=0)

    rng = np.random.default_rng(seed)
    count = np.zeros(diffs.shape[1], dtype=np.int64)
    for start in range(0, n_perm, PERM_CHUNK):
        size = min(PERM_CHUNK, n_perm - start)
        signs = rng.choice([-1.0, 1.0], size=(size, n_part))
        count += (np.abs(signs @ totals) >= np.abs(obs) - 1e-9).sum(axis=0)
    return obs / len(diffs), (count + 1) / (n_perm + 1)


def omnibus(frame, sources, n_perm=N_PERM, seed=SEED):
    """Permutation test that all `sources` have the same MAE.

    Statistic: the variance of the sources' MAEs. Null distribution: within
    each participant the source labels on the absolute errors are permuted,
    one permutation per participant for all of their cases. Returns
    {statistic, p}; p is (exceedances + 1) / (n_perm + 1), never exactly 0.
    """
    h, p, _ = _arrays(frame, sources)
    err = np.abs(h[:, None] - p)
    codes = participant_codes(frame)
    totals = np.zeros((codes.max() + 1, err.shape[1]))
    np.add.at(totals, codes, err)
    n = len(err)
    obs = float(np.var(totals.sum(axis=0) / n))

    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, n_perm, PERM_CHUNK):
        size = min(PERM_CHUNK, n_perm - start)
        shuffled = rng.permuted(np.broadcast_to(totals, (size, *totals.shape)), axis=2)
        count += int((np.var(shuffled.sum(axis=1) / n, axis=1) >= obs - 1e-12).sum())
    return {"statistic": obs, "p": (count + 1) / (n_perm + 1)}


def pairwise(frame, sources, mae_reps, n_perm=N_PERM, seed=SEED, alpha=ALPHA):
    """Every pair of sources on MAE, a before b in `sources` order.

    diff = MAE(a) - MAE(b); negative means a is closer to the humans. The CI
    comes from the bootstrap replicates `mae_reps` (n_boot, S), columns in
    `sources` order; the p-value from cluster_sign_flip, unadjusted. A pair is
    significant when p < alpha / number of pairs (Bonferroni).
    """
    h, p, _ = _arrays(frame, sources)
    err = np.abs(h[:, None] - p)
    pairs = list(itertools.combinations(range(len(sources)), 2))
    diffs = np.column_stack([err[:, a] - err[:, b] for a, b in pairs])
    obs, pvals = cluster_sign_flip(diffs, participant_codes(frame), n_perm, seed)
    threshold = alpha / len(pairs)

    rows = []
    for k, (a, b) in enumerate(pairs):
        lo, hi = _ci(mae_reps[:, a] - mae_reps[:, b])
        rows.append({"a": sources[a], "b": sources[b], "diff": float(obs[k]),
                     "ci": [float(lo), float(hi)], "p": float(pvals[k]),
                     "significant": bool(pvals[k] < threshold)})
    return rows


def letter_display(names, pairs):
    """Compact letter display: sources sharing a letter do not differ significantly.

    names -- display order, best first
    pairs -- rows with a, b, significant (as pairwise returns)

    Each letter is a maximal set of mutually non-different sources (a maximal
    clique of the "not significant" graph), so every non-significant pair
    shares a letter and no significant pair does. Letters run in order of the
    best member of each set.
    """
    same = {n: set() for n in names}
    for r in pairs:
        if not r["significant"]:
            same[r["a"]].add(r["b"])
            same[r["b"]].add(r["a"])

    cliques = []

    def expand(clique, candidates, excluded):          # Bron-Kerbosch
        if not candidates and not excluded:
            cliques.append(clique)
            return
        for v in list(candidates):
            expand(clique | {v}, candidates & same[v], excluded & same[v])
            candidates = candidates - {v}
            excluded = excluded | {v}

    expand(set(), set(names), set())
    pos = {n: i for i, n in enumerate(names)}
    cliques.sort(key=lambda c: (min(pos[n] for n in c), max(pos[n] for n in c)))
    letters = {n: "" for n in names}
    for letter, clique in zip(string.ascii_lowercase, cliques):
        for n in clique:
            letters[n] += letter
    return letters


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def compute(frame, llms, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED, alpha=ALPHA,
            baselines=BASELINES):
    llms = list(llms)
    sources = llms + [b for b in baselines if b in frame.columns]
    h, p, init = _arrays(frame, sources)
    point = {k: v[0] for k, v in score(np.ones((1, len(h))), h, p, init).items()}
    reps = bootstrap(frame, sources, n_boot, seed)
    table = summarize(point, reps, sources, llms)

    order = sorted(sources, key=lambda n: table[PRIMARY][n]["estimate"])
    idx = [sources.index(n) for n in order]
    pairs = pairwise(frame, order, reps[PRIMARY][:, idx], n_perm, seed, alpha)
    return {
        "n_obs": int(len(frame)),
        "n_participants": int(frame["persona_id"].nunique()),
        "n_boot": n_boot, "n_perm": n_perm, "seed": seed, "alpha": alpha,
        "alpha_pair": alpha / len(pairs) if pairs else alpha,
        "llms": llms, "baselines": [b for b in sources if b not in llms],
        "order": order, "metrics": table,
        "omnibus": omnibus(frame, llms, n_perm, seed),
        "pairwise": pairs,
        "letters": letter_display(order, pairs),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _f(x, spec="+.3f"):
    return "inf" if np.isinf(x) else format(x, spec)


def _ci_text(ci, spec=".3f"):
    return f"[{_f(ci[0], spec)}, {_f(ci[1], spec)}]"


def render(r):
    lines = []
    out = lines.append
    llms, baselines = r["llms"], r["baselines"]

    out(RULE)
    out("Ranking the LLMs by how faithfully they simulate the humans")
    out(f"n = {r['n_obs']} persona x topic cases from {r['n_participants']} participants; "
        f"participant-level bootstrap, {r['n_boot']} replicates, seed {r['seed']}")
    if r.get("dropped") and any(r["dropped"].values()):
        out(f"cases dropped for all sources because a model's response did not parse: "
            f"{ {k: v for k, v in r['dropped'].items() if v} }")
    if baselines:
        out(f"baselines (scored, not ranked): {', '.join(baselines)}")
    out(RULE)

    for metric, description in METRICS.items():
        rows = r["metrics"][metric]
        out("")
        out(f"{metric.upper()}  {description}   lower is better"
            + ("   [primary]" if metric == PRIMARY else ""))
        out(f"{'source':<26}{'estimate':>10}{'95% CI':>22}{'rank':>7}{'rank CI':>11}"
            f"{'P(best)':>10}")
        out(LINE)
        for name in sorted(rows, key=lambda n: rows[n]["estimate"]):
            row = rows[name]
            text = f"{name:<26}{_f(row['estimate'], '.4f'):>10}{_ci_text(row['ci'], '.4f'):>22}"
            if name in llms:
                rank_ci = f"{row['rank_ci'][0]}-{row['rank_ci'][1]}"
                text += f"{row['rank']:>7}{rank_ci:>11}{row['p_best']:>10.3f}"
            else:
                text += f"{'-':>7}{'-':>11}{'-':>10}"
            out(text)
        out(LINE)
    out("rank, rank CI and P(best) are among the LLMs only; ties share the better rank")

    out("")
    out("BIAS  mean(human - model) post-stance; > 0: humans end more pro-topic "
        "(tested by the permutation test)")
    out(f"{'source':<26}{'bias':>10}{'95% CI':>22}")
    out(LINE)
    for name in r["order"]:
        row = r["metrics"]["bias"][name]
        out(f"{name:<26}{_f(row['estimate'], '+.4f'):>10}{_ci_text(row['ci'], '+.4f'):>22}")
    out(LINE)

    out("")
    out(RULE)
    out(f"Omnibus: do the {len(llms)} LLMs differ in MAE at all?")
    out("statistic = variance of the LLMs' MAEs; null: model labels permuted within "
        f"each participant, {r['n_perm']} draws")
    out(RULE)
    out(f"statistic = {r['omnibus']['statistic']:.6f}   p = {r['omnibus']['p']:.2e}")

    out("")
    out(RULE)
    out("Pairwise MAE differences: diff = MAE(a) - MAE(b), < 0 means a is closer to "
        "the humans")
    out(f"cluster sign-flip permutation test (whole participants flipped), "
        f"{r['n_perm']} draws, two-sided; raw p;")
    out(f"sig = Bonferroni across all {len(r['pairwise'])} pairs, "
        f"p < {r['alpha']}/{len(r['pairwise'])} = {r['alpha_pair']:.5f}")
    out(RULE)
    out(f"{'a':<24}{'b':<24}{'diff':>9}{'95% CI':>20}{'p':>11}  sig")
    out(LINE)
    for row in r["pairwise"]:
        out(f"{row['a']:<24}{row['b']:<24}{row['diff']:>+9.4f}"
            f"{_ci_text(row['ci'], '+.3f'):>20}{row['p']:>11.2e}"
            f"  {'yes' if row['significant'] else 'no'}")
    out(LINE)

    out("")
    out("Tiers on MAE (compact letter display): sources sharing a letter are not "
        "significantly different")
    out(f"{'source':<26}{'MAE':>10}   tier")
    out(LINE)
    for name in r["order"]:
        out(f"{name:<26}{r['metrics'][PRIMARY][name]['estimate']:>10.4f}   "
            f"{r['letters'][name]}")
    out(LINE)
    if baselines:
        out(f"{NO_CHANGE}: predicts the initial belief. {CELL_MEDIAN}: median post-stance "
            "of the other humans")
        out("in the same topic x package x initial-belief cell. A model that does not "
            "beat the cell median")
        out("predicts the individual no better than 'someone like you'.")
    return "\n".join(lines) + "\n"


def _label(name):
    return MODEL_LABELS.get(name, name)


def render_tex(r):
    """One row per pair; bold p below Bonferroni. The omnibus test is reported
    in the SI text, not here."""
    n_pairs = len(r["pairwise"])
    draws = f"{r['n_perm']:,}".replace(",", "{,}")
    boots = f"{r['n_boot']:,}".replace(",", "{,}")
    mae = r["metrics"][PRIMARY]

    lines = []
    out = lines.append
    out(r"% Generated by scripts.stats.model_ranking -- do not edit by hand.")
    out(r"% Regenerate:  python -m scripts.stats.model_ranking")
    out(r"% Requires \usepackage{booktabs}")
    out(r"\begin{table}[ht]")
    out(r"\centering")
    out(r"\caption{Pairwise comparison of the LLMs' mean absolute error (MAE) "
        r"relative to the human post-stance, over " + f"{r['n_obs']:,}".replace(",", "{,}")
        + r" participant-topic cases from " + str(r["n_participants"])
        + r" participants. $\Delta_{\mathrm{MAE}}$ is "
        r"MAE(A) $-$ MAE(B); negative values mean that A is closer to the humans. "
        r"95\% CIs are participant-level cluster bootstrap percentile intervals ("
        + boots + r" replicates). $p$-values are uncorrected and come from two-sided "
        r"paired sign-flip permutation tests that flip whole participants ("
        + draws + r" permutations). Bold $p$-values are significant after Bonferroni "
        r"correction across all " + str(n_pairs) + r" pairs ($p < 0.05/"
        + str(n_pairs) + r"$).}")
    out(r"\label{tab:si_model_ranking_pairwise}")
    out(r"\small")
    out(r"\begin{tabular}{ll rrr r}")
    out(r"\toprule")
    out(r"Model A & Model B & MAE(A) & MAE(B) & $\Delta_{\mathrm{MAE}}$ [95\% CI] & $p$ \\")
    out(r"\midrule")
    first = None
    for row in r["pairwise"]:
        if first is not None and row["a"] != first:
            out(r"\addlinespace")
        first = row["a"]
        ci = f"[{num(row['ci'][0])}, {num(row['ci'][1])}]"
        out(" & ".join([_label(row["a"]), _label(row["b"]),
                        num(mae[row["a"]]["estimate"]), num(mae[row["b"]]["estimate"]),
                        f"{num(row['diff'])} {ci}",
                        p_cell(row["p"], r["alpha_pair"])]) + r" \\")
    out(r"\bottomrule")
    out(r"\end{tabular}")
    out(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--baselines", action="store_true",
                        help="also score the no-change and cell-median baselines "
                             "and include them in the pairwise tests")
    args = parser.parse_args()

    frame, dropped = load()
    r = compute(frame, list(MODELS), args.n_boot, args.n_perm, args.seed,
                baselines=BASELINES if args.baselines else ())
    r["dropped"] = dropped
    text = render(r)
    print(text)

    ensure_output_dirs()
    outputs = {"model_ranking.txt": text,
               "model_ranking.json": json.dumps(r, indent=2),
               "model_ranking.tex": render_tex(r)}
    for filename, content in outputs.items():
        (STATS_OUTPUT_DIR / filename).write_text(content, encoding="utf-8")
        print(f"wrote {STATS_OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
