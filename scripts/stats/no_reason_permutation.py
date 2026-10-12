"""Paired permutation tests, base prompt vs the `no-reason` ablation, as LaTeX.

Does asking an LLM to write a reason for the simulated belief change affect
the belief update it simulates? The `no-reason` ablation asks for the same JSON
minus the "reasoning" field; everything else about the prompt is unchanged, and
Qwen3-32B still thinks before answering in both settings.

For each model and setting, the human post-stance is compared with the model's
exactly as in the main analysis -- Delta, its sign-flip permutation p, and MAE
-- over all topics and within each topic (no_reason_permutation.tex).

A second table (no_reason_direct.tex) tests the two settings against each
other on the same participant-topic pairs, again by paired sign-flip:

    stance shift     base post-stance - no-reason post-stance
    MAE difference   |human - base| - |human - no reason|
                     (> 0: dropping the reason moved the model closer to humans)

    python -m scripts.stats.no_reason_permutation             # ~2 min, 32 tests
    python -m scripts.stats.no_reason_permutation --rebuild   # instant, from JSON
"""

import argparse
import json

from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data
from belief_update_sim.permutation_test import PERMUTATION_RESAMPLES, fast_permutation_test

from scripts.stats.permutation_by_topic import (TOPIC_LABELS, cell, num, p_cell,
                                                paired, triple)

JSON_PATH = STATS_OUTPUT_DIR / "no_reason_permutation.json"
TEX_PATH = STATS_OUTPUT_DIR / "no_reason_permutation.tex"
DIRECT_TEX_PATH = STATS_OUTPUT_DIR / "no_reason_direct.tex"

# model label -> {setting: results file}, in table order
RUNS = {
    "Llama-3.3-70B-Instruct": {
        "base": "Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "no_reason": "ablations/Llama-3.3-70B-Instruct_no-reason.xlsx"},
    "Qwen3-32B": {
        "base": "Qwen3_32B_all_personas_results.xlsx",
        "no_reason": "ablations/Qwen3-32B_no-reason.xlsx"},
}
SETTING_LABELS = {"base": "Base", "no_reason": "No reason"}

# column scopes: None is every topic together
SCOPES = {"all": None, **{t: t for t in TOPIC_LABELS}}
SCOPE_LABELS = {"all": "All topics", **TOPIC_LABELS}


def compute(n_resamples=PERMUTATION_RESAMPLES):
    """{model: {setting: {scope: cell}}}"""
    human = load_human_normalized_data()
    results = {}
    for name, settings in RUNS.items():
        results[name] = {}
        for setting, filename in settings.items():
            model = load_normalized_data(RESULTS_DIR / filename)
            results[name][setting] = {}
            for scope, topic in SCOPES.items():
                h = human if topic is None else human[human["topic"] == topic]
                c = cell(*paired(h, model, f"{name} {setting}"), n_resamples)
                results[name][setting][scope] = c
                print(f"{name:<24} {setting:<10} {scope:<12} n={c['n']}", flush=True)
    return {"n_resamples": n_resamples, "results": results,
            "direct": compute_direct(human, n_resamples)}


def triples(human, base, ablation, name=""):
    """Human, base and no-reason post-stance on the pairs all three answered.

    All three must agree on the comment package, or the two model runs (or
    the humans) would not have read the same comments.
    """
    keys = ["persona_id", "topic"]
    frames = [df[keys + ["package", "new_belief"]].rename(
                  columns={"package": f"package_{tag}", "new_belief": tag})
              for df, tag in ((human, "human"), (base, "base"), (ablation, "no_reason"))]
    merged = (frames[0].merge(frames[1], on=keys, validate="one_to_one")
                       .merge(frames[2], on=keys, validate="one_to_one"))
    mismatched = ((merged["package_human"] != merged["package_base"])
                  | (merged["package_human"] != merged["package_no_reason"]))
    if mismatched.any():
        raise ValueError(f"{name}: {int(mismatched.sum())} participant-topics "
                         "were shown different comment packages across the runs.")
    return merged.dropna(subset=["human", "base", "no_reason"])


def direct_cell(t, n_resamples=PERMUTATION_RESAMPLES):
    """Base vs no-reason on one scope: stance shift and MAE difference."""
    err_base = (t["human"] - t["base"]).abs()
    err_ablation = (t["human"] - t["no_reason"]).abs()
    shift, p_shift, _ = fast_permutation_test(t["base"], t["no_reason"], n_resamples)
    mae_diff, p_mae, _ = fast_permutation_test(err_base, err_ablation, n_resamples)
    return {"n": int(len(t)),
            "agree": float((t["base"] == t["no_reason"]).mean()),
            "shift": float(shift), "p_shift": float(p_shift),
            "mae_base": float(err_base.mean()),
            "mae_no_reason": float(err_ablation.mean()),
            "mae_diff": float(mae_diff), "p_mae": float(p_mae)}


def compute_direct(human, n_resamples=PERMUTATION_RESAMPLES):
    """{model: {scope: direct_cell}}"""
    direct = {}
    for name, settings in RUNS.items():
        t = triples(human,
                    load_normalized_data(RESULTS_DIR / settings["base"]),
                    load_normalized_data(RESULTS_DIR / settings["no_reason"]), name)
        direct[name] = {}
        for scope, topic in SCOPES.items():
            sub = t if topic is None else t[t["topic"] == topic]
            direct[name][scope] = direct_cell(sub, n_resamples)
            print(f"{name:<24} direct     {scope:<12} n={len(sub)}", flush=True)
    return direct


def render(payload):
    results = payload["results"]
    scopes = list(SCOPES)
    n_tests = sum(len(settings) * len(scopes) for settings in results.values())
    alpha = 0.05 / n_tests
    resamples = f"{payload['n_resamples']:,}".replace(",", "{,}")
    n_columns = 1 + 3 * len(scopes)
    width = max(len(v) for v in SETTING_LABELS.values())

    lines = []
    out = lines.append
    out(r"% Generated by scripts.stats.no_reason_permutation -- do not edit by hand.")
    out(r"% Regenerate:  python -m scripts.stats.no_reason_permutation --rebuild")
    out(r"% Requires \usepackage{booktabs}")
    out(r"\begin{table}[ht]")
    out(r"\centering")
    out(r"\caption{Paired permutation test results with and without a stated "
        r"reason for the belief change. \emph{Base} is the main prompt, which "
        r"asks the model for its simulated reasoning alongside the new belief; "
        r"\emph{No reason} asks for the same output without the reasoning "
        r"field. Each setting is compared with the human participants "
        r"separately. $\Delta$ denotes the mean paired final stance difference "
        r"(human post-stance minus matched LLM post-stance); MAE is the mean "
        r"absolute paired difference, which unlike $\Delta$ cannot cancel "
        r"opposing errors. $p$-values test $\Delta = 0$ and are based on "
        + resamples + r" sign-flip resamples; MAE is descriptive. Bold "
        r"$p$-values are significant after Bonferroni correction across all "
        + str(n_tests) + r" tests ($p < 0.05/" + str(n_tests) + r"$).}")
    out(r"\label{tab:si_permutation_no_reason}")
    out(r"\footnotesize")
    out(r"\setlength{\tabcolsep}{4pt}")
    out(r"\begin{tabular}{l" + " rrr" * len(scopes) + "}")
    out(r"\toprule")
    out(" & " + " & ".join(r"\multicolumn{3}{c}{" + SCOPE_LABELS[s] + "}"
                           for s in scopes) + r" \\")
    out(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                 for i in range(len(scopes))))
    out("Setting & " + " & ".join([r"$\Delta$ & $p$ & MAE"] * len(scopes)) + r" \\")
    out(r"\midrule")
    for i, (name, settings) in enumerate(results.items()):
        if i:
            out(r"\addlinespace")
        out(r"\multicolumn{" + str(n_columns) + r"}{l}{\textit{" + name + r"}} \\")
        for setting in SETTING_LABELS:
            if setting not in settings:
                continue
            cells = [x for s in scopes for x in triple(settings[setting][s], alpha)]
            out(f"{SETTING_LABELS[setting]:<{width}} & " + " & ".join(cells) + r" \\")
    out(r"\bottomrule")
    out(r"\end{tabular}")
    out(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_direct(payload):
    """Scopes down the side under a heading row per model, statistics across."""
    direct = payload["direct"]
    n_tests = 2 * sum(len(scopes) for scopes in direct.values())
    alpha = 0.05 / n_tests
    resamples = f"{payload['n_resamples']:,}".replace(",", "{,}")
    width = max(len(v) for v in SCOPE_LABELS.values())

    lines = []
    out = lines.append
    out(r"% Generated by scripts.stats.no_reason_permutation -- do not edit by hand.")
    out(r"% Regenerate:  python -m scripts.stats.no_reason_permutation --rebuild")
    out(r"% Requires \usepackage{booktabs}")
    out(r"\begin{table}[ht]")
    out(r"\centering")
    out(r"\caption{Direct comparison of the base prompt with the \emph{No "
        r"reason} ablation on the same participant-topic pairs. \emph{Agree} "
        r"is the share of pairs on which both settings give the same "
        r"post-stance. $\Delta_{\mathrm{stance}}$ is the mean paired "
        r"difference in simulated post-stance (base minus no reason). The two "
        r"MAE columns are each setting's mean absolute difference from the "
        r"human post-stance, and $\Delta_{\mathrm{MAE}}$ is their paired "
        r"difference (base minus no reason; positive means dropping the reason "
        r"moved the simulation closer to the humans). Both $p$-values are from "
        r"two-sided paired sign-flip permutation tests with " + resamples
        + r" resamples, testing $\Delta_{\mathrm{stance}} = 0$ and "
        r"$\Delta_{\mathrm{MAE}} = 0$ respectively. Bold $p$-values are "
        r"significant after Bonferroni correction across all " + str(n_tests)
        + r" tests ($p < 0.05/" + str(n_tests) + r"$).}")
    out(r"\label{tab:si_no_reason_direct}")
    out(r"\small")
    out(r"\begin{tabular}{l rrr rrrr}")
    out(r"\toprule")
    out(r" & & \multicolumn{2}{c}{Stance shift} & \multicolumn{4}{c}{Fit to humans} \\")
    out(r"\cmidrule(lr){3-4} \cmidrule(lr){5-8}")
    out(r"Topic & Agree & $\Delta_{\mathrm{stance}}$ & $p$ & MAE base "
        r"& MAE no reason & $\Delta_{\mathrm{MAE}}$ & $p$ \\")
    out(r"\midrule")
    for i, (name, scopes) in enumerate(direct.items()):
        if i:
            out(r"\addlinespace")
        out(r"\multicolumn{8}{l}{\textit{" + name + r"}} \\")
        for scope in SCOPES:
            c = scopes[scope]
            cells = [f"{100 * c['agree']:.1f}" + r"\%",
                     num(c["shift"]), p_cell(c["p_shift"], alpha),
                     num(c["mae_base"]), num(c["mae_no_reason"]),
                     num(c["mae_diff"]), p_cell(c["p_mae"], alpha)]
            out(f"{SCOPE_LABELS[scope]:<{width}} & " + " & ".join(cells) + r" \\")
    out(r"\bottomrule")
    out(r"\end{tabular}")
    out(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true",
                        help=f"render the table from {JSON_PATH.name} instead of "
                             "recomputing the permutation tests")
    args = parser.parse_args()

    ensure_output_dirs()
    if args.rebuild:
        try:
            payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SystemExit(f"no results at {JSON_PATH}; run without --rebuild first")
        if "direct" not in payload:
            raise SystemExit(f"{JSON_PATH} predates the direct comparison; "
                             "run without --rebuild to recompute it")
    else:
        payload = compute()
        JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {JSON_PATH}")

    for path, text in ((TEX_PATH, render(payload)),
                       (DIRECT_TEX_PATH, render_direct(payload))):
        print(text)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
