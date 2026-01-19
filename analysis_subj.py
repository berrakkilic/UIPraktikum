import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------
# CONFIG: input files
# -----------------------------
FILES = {
    "tick-note": "tick-note.csv",
    "add-plant": "add-plant.csv",
    "sign-up": "sign-up.csv",
    "kiwi-favourites": "kiwi-favourites.csv",
}

OUT_DIR = "ab_outputs_subjective"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def find_click_col(df: pd.DataFrame):
    # not strictly needed for subjective, but used for excluding objective cols robustly
    for c in ["clicks", "number_of_clicks", "number_of_click", "click_count", "click"]:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if "click" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    # if none or ambiguous, just return None
    return None


def detect_subjective_cols(df: pd.DataFrame):
    """
    Subjective cols = numeric cols excluding ID + condition + objective columns.
    """
    click_col = find_click_col(df)

    exclude = {
        "participant_id", "condition",
        "reaction_time",
        "major_errors", "minor_erros", "minor_errors",  # tolerate both spellings
    }

    if click_col:
        exclude.add(click_col)

    # if error_score exists (or can be computed), exclude it too
    if "error_score" in df.columns:
        exclude.add("error_score")

    subjective = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            subjective.append(c)

    return subjective


def paired_wide(df: pd.DataFrame, col: str):
    wide = df.pivot(index="participant_id", columns="condition", values=col)
    # keep only complete A/B pairs
    wide = wide.dropna(subset=["A", "B"])
    return wide


def save_hist(diffs, title, path):
    fig, ax = plt.subplots()
    ax.hist(diffs, bins="auto")
    ax.set_title(title)
    ax.set_xlabel("Difference (B - A)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_qq(diffs, title, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sm.ProbPlot(diffs, dist=stats.norm).qqplot(line="45", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_box_AB(A, B, title, path):
    fig, ax = plt.subplots()
    ax.boxplot([A, B], labels=["A", "B"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_homoscedasticity_plot(A, B, title, path):
    """
    Paired-friendly spread check:
    plots |B-A| vs average (A+B)/2.
    Fan shape => heteroscedasticity (variance changes with magnitude).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    avg = (A + B) / 2
    abs_diff = np.abs(B - A)

    fig, ax = plt.subplots()
    ax.scatter(avg, abs_diff)
    ax.set_title(title)
    ax.set_xlabel("Average of conditions (A+B)/2")
    ax.set_ylabel("Absolute difference |B-A|")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def analyze_subjective(task_name: str, wide: pd.DataFrame, item_col: str):
    A = wide["A"].to_numpy()
    B = wide["B"].to_numpy()
    d = B - A
    n = len(d)
    dfree = n - 1

    # Normality (differences)
    shapiro_W, shapiro_p = stats.shapiro(d) if n >= 3 else (np.nan, np.nan)

    # Homoscedasticity (A vs B) - as requested by slides
    lev_stat, lev_p = stats.levene(A, B, center="median") if n >= 2 else (np.nan, np.nan)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(B, A) if n >= 2 else (np.nan, np.nan)

    # Effect size: Cohen's dz
    mean_d = float(np.mean(d)) if n > 0 else np.nan
    sd_d = float(np.std(d, ddof=1)) if n > 1 else np.nan
    cohen_dz = mean_d / sd_d if (n > 1 and sd_d != 0) else np.nan

    # 95% CI for mean difference
    if n > 1 and not np.isnan(sd_d):
        se = sd_d / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, dfree)
        ci_low = mean_d - tcrit * se
        ci_high = mean_d + tcrit * se
    else:
        ci_low, ci_high = np.nan, np.nan

    # Plots
    safe_item = item_col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    base = f"{task_name}__{safe_item}"

    save_hist(
        d,
        f"{task_name} – {item_col}: Hist of diffs (B-A)",
        os.path.join(PLOTS_DIR, f"{base}__hist.png"),
    )
    save_qq(
        d,
        f"{task_name} – {item_col}: Q–Q of diffs (B-A)",
        os.path.join(PLOTS_DIR, f"{base}__qq.png"),
    )
    save_box_AB(
        A,
        B,
        f"{task_name} – {item_col}: A vs B spread",
        os.path.join(PLOTS_DIR, f"{base}__box.png"),
    )
    save_homoscedasticity_plot(
        A, B,
        f"{task_name} – {item_col}: |B-A| vs avg (spread check)",
        os.path.join(PLOTS_DIR, f"{base}__homo.png"),
    )

    return {
        "task": task_name,
        "subjective_item": item_col,
        "n_pairs": int(n),
        "mean_A": float(np.mean(A)) if n > 0 else np.nan,
        "mean_B": float(np.mean(B)) if n > 0 else np.nan,
        "mean_diff_B_minus_A": mean_d,
        "shapiro_W_diff": float(shapiro_W) if not np.isnan(shapiro_W) else np.nan,
        "shapiro_p_diff": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "levene_stat_AB": float(lev_stat) if not np.isnan(lev_stat) else np.nan,
        "levene_p_AB": float(lev_p) if not np.isnan(lev_p) else np.nan,
        "t": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "df": int(dfree) if dfree >= 0 else np.nan,
        "p": float(p_val) if not np.isnan(p_val) else np.nan,
        "cohen_dz": float(cohen_dz) if not np.isnan(cohen_dz) else np.nan,
        "ci95_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
        "ci95_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
    }


# -----------------------------
# Main
# -----------------------------
all_rows = []
skipped = []

for task, filename in FILES.items():
    if not os.path.exists(filename):
        skipped.append((task, filename, "file not found"))
        continue

    df = pd.read_csv(filename)

    # Basic required columns
    if not {"participant_id", "condition"}.issubset(df.columns):
        skipped.append((task, filename, "missing participant_id or condition"))
        continue

    # Optional: compute error_score if possible (only to exclude it from subjective detection)
    if "error_score" not in df.columns:
        if "major_errors" in df.columns and ("minor_erros" in df.columns or "minor_errors" in df.columns):
            minor_col = "minor_erros" if "minor_erros" in df.columns else "minor_errors"
            df["error_score"] = df["major_errors"] + 0.5 * df[minor_col]

    subjective_cols = detect_subjective_cols(df)

    if not subjective_cols:
        skipped.append((task, filename, "no subjective numeric columns detected"))
        continue

    for col in subjective_cols:
        wide = paired_wide(df, col)
        if len(wide) < 2:
            skipped.append((task, filename, f"{col}: not enough A/B pairs"))
            continue

        row = analyze_subjective(task, wide, col)
        all_rows.append(row)

results = pd.DataFrame(all_rows).sort_values(["task", "subjective_item"])
os.makedirs(OUT_DIR, exist_ok=True)
out_csv = os.path.join(OUT_DIR, "paired_results_subjective.csv")
results.to_csv(out_csv, index=False)

print("✅ DONE")
print("Subjective results saved to:", out_csv)
print("Plots saved to:", PLOTS_DIR)

if skipped:
    print("\n--- Skipped items (so you know what happened) ---")
    for s in skipped:
        print(" -", s)
