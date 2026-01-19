import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---- input files ----
FILES = {
    "tick-note": "tick-note.csv",
    "add-plant": "add-plant.csv",
    "sign-up": "sign-up.csv",
    "kiwi-favourites": "kiwi-favourites.csv",
}

OUT_DIR = "ab_outputs_objective"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_click_col(df):
    # different tasks name the click column differently
    for c in ["clicks", "number_of_clicks", "number_of_click", "click_count", "click"]:
        if c in df.columns:
            return c
    # fallback: any column containing "click"
    candidates = [c for c in df.columns if "click" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not identify click column. Found: {candidates}")

def paired_table(df, metric_col):
    # expects columns: participant_id, condition, metric_col
    wide = df.pivot(index="participant_id", columns="condition", values=metric_col)
    # keep only complete pairs
    wide = wide.dropna(subset=["A", "B"])
    return wide

def qqplot(data, title, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sm.ProbPlot(data, dist=stats.norm).qqplot(line="45", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def histplot(data, title, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins="auto")
    ax.set_title(title)
    ax.set_xlabel("Difference (B - A)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def boxplot_AB(wide, title, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([wide["A"].values, wide["B"].values], labels=["A", "B"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def homoscedasticity_plot(A, B, title, path):
    """
    Paired-friendly heteroscedasticity check:
    plots |B-A| against the average (A+B)/2.
    If spread grows with the mean, variance is not constant.
    """
    avg = (A + B) / 2
    abs_diff = np.abs(B - A)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(avg, abs_diff)
    ax.set_title(title)
    ax.set_xlabel("Average of conditions (A+B)/2")
    ax.set_ylabel("Absolute difference |B-A|")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def analyze_metric(task_name, wide, metric_name):
    A = wide["A"].to_numpy()
    B = wide["B"].to_numpy()
    d = B - A
    n = len(d)

    # ---- normality on differences ----
    shapiro_W, shapiro_p = stats.shapiro(d)  # optional, but included

    # plots
    base = f"{task_name}__{metric_name}".replace("/", "_")
    histplot(d, f"{task_name} – {metric_name}: Hist of diffs (B-A)",
             os.path.join(PLOTS_DIR, f"{base}__hist.png"))
    qqplot(d, f"{task_name} – {metric_name}: Q–Q of diffs (B-A)",
           os.path.join(PLOTS_DIR, f"{base}__qq.png"))

    # ---- homoscedasticity A vs B (Levene) ----
    lev_stat, lev_p = stats.levene(A, B, center="median")

    homoscedasticity_plot(
        A, B,
        f"{task_name} – {metric_name}: |B-A| vs avg (spread check)",
        os.path.join(PLOTS_DIR, f"{base}__homo.png")
    )
    
    boxplot_AB(wide, f"{task_name} – {metric_name}: A vs B spread",
               os.path.join(PLOTS_DIR, f"{base}__box.png"))

    # ---- paired t-test ----
    t_stat, p_val = stats.ttest_rel(B, A)  # tests mean(B-A) = 0
    df = n - 1
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    cohen_dz = mean_d / sd_d if sd_d != 0 else np.nan

    # 95% CI for mean diff
    se = sd_d / np.sqrt(n) if n > 0 else np.nan
    t_crit = stats.t.ppf(0.975, df) if df > 0 else np.nan
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se

    return {
        "task": task_name,
        "metric": metric_name,
        "n_pairs": n,
        "mean_A": float(np.mean(A)),
        "mean_B": float(np.mean(B)),
        "mean_diff_B_minus_A": mean_d,
        "shapiro_W_diff": float(shapiro_W),
        "shapiro_p_diff": float(shapiro_p),
        "levene_stat_AB": float(lev_stat),
        "levene_p_AB": float(lev_p),
        "t": float(t_stat),
        "df": int(df),
        "p": float(p_val),
        "cohen_dz": float(cohen_dz),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }

all_results = []

for task, path in FILES.items():
    df = pd.read_csv(path)

    # compute error_score
    if not {"major_errors", "minor_erros"}.issubset(df.columns):
        raise ValueError(f"{task}: missing major_errors or minor_erros columns")
    df["error_score"] = df["major_errors"] + 0.5 * df["minor_erros"]

    click_col = find_click_col(df)

    metrics = {
        "reaction_time_ms": "reaction_time",
        "clicks": click_col,
        "error_score": "error_score",
    }

    for metric_name, col in metrics.items():
        wide = paired_table(df, col)
        res = analyze_metric(task, wide, metric_name)
        all_results.append(res)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "paired_results_objective.csv"), index=False)

print("Done.")
print("Saved plots to:", PLOTS_DIR)
print("Saved results to:", os.path.join(OUT_DIR, "paired_results_objective.csv"))
