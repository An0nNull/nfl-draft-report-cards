"""
NFL Draft Analysis — Team Drafting Efficiency
Reads data/draft_data.csv, outputs team scores, all picks, and expected pick curve.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

INPUT_FILE = "data/draft_data.csv"
OUTPUT_DIR = "data"
EXTRA_DRAFT_FILES = {
    2023: r"C:\Users\jay58\Downloads\2023NFL_Draft_sportsref_download.csv",
    2024: r"C:\Users\jay58\Downloads\2024NFL_Draft_sportsref_download.csv",
}
TEAM_OUT   = os.path.join(OUTPUT_DIR, "team_scores.csv")
TEAM_JSON  = os.path.join(OUTPUT_DIR, "team_scores.json")
CURVE_OUT  = os.path.join(OUTPUT_DIR, "pick_curve.csv")
PICKS_OUT  = os.path.join(OUTPUT_DIR, "all_picks.csv")
STEALS_OUT = os.path.join(OUTPUT_DIR, "top_steals.csv")

POS_GROUPS = {
    "QB":   ["QB"],
    "RB":   ["RB", "FB"],
    "WR":   ["WR"],
    "TE":   ["TE"],
    "OL":   ["OT", "OG", "C", "OL", "G", "T"],
    "DL":   ["DT", "DE", "NT", "DL"],
    "EDGE": ["OLB", "EDGE"],
    "LB":   ["ILB", "MLB", "LB"],
    "CB":   ["CB"],
    "S":    ["S", "SS", "FS"],
    "ST":   ["K", "P", "LS"],
}

def map_pos_group(pos):
    if not isinstance(pos, str):
        return "OTHER"
    for group, positions in POS_GROUPS.items():
        if pos.upper() in positions:
            return group
    return "OTHER"

def power_decay(x, a, b, c):
    return a * np.power(x, -b) + c

def fit_expected_curve(df):
    pick_stats = (
        df.groupby("pick")["career_av"]
        .agg(["median", "mean", "count"])
        .reset_index()
    )
    pick_stats = pick_stats[pick_stats["count"] >= 3]
    x = pick_stats["pick"].values
    y = pick_stats["median"].values

    try:
        popt, _ = curve_fit(
            power_decay, x, y,
            p0=[50, 0.5, 2],
            bounds=([0, 0.01, 0], [500, 2, 20]),
            maxfev=10000
        )
        print(f"Curve fit: a={popt[0]:.2f}, b={popt[1]:.3f}, c={popt[2]:.2f}")
        fit_fn = lambda p: power_decay(p, *popt)
    except Exception as e:
        print(f"Curve fit failed ({e}), using interpolation.")
        from scipy.interpolate import interp1d
        fit_fn = interp1d(x, y, kind="linear", fill_value="extrapolate")

    picks = np.arange(1, 263)
    return pd.Series(
        [max(0, float(fit_fn(p))) for p in picks],
        index=picks,
        name="expected_av"
    )

def load_extra_draft_file(year, path):
    if not os.path.exists(path):
        return pd.DataFrame()

    extra = pd.read_csv(path, skiprows=1)
    extra = extra.rename(columns={
        "Rnd": "round",
        "Pick": "pick",
        "Tm": "team",
        "Player": "pfr_player_name",
        "Pos": "position",
        "College/Univ": "college",
        "wAV": "w_av",
        "G": "games",
        "PB": "probowls",
        "AP1": "allpro",
    })
    extra["season"] = year

    cols = [
        "season", "round", "pick", "team", "pfr_player_name", "position",
        "college", "w_av", "games", "probowls", "allpro"
    ]
    return extra[[c for c in cols if c in extra.columns]]

def grade_score(score, all_scores):
    spread = all_scores.max() - all_scores.min()
    pct = 0.5 if spread == 0 else (score - all_scores.min()) / spread
    if pct >= 0.85: return "A+"
    if pct >= 0.75: return "A"
    if pct >= 0.65: return "A-"
    if pct >= 0.55: return "B+"
    if pct >= 0.45: return "B"
    if pct >= 0.35: return "B-"
    if pct >= 0.25: return "C+"
    if pct >= 0.18: return "C"
    if pct >= 0.12: return "C-"
    if pct >= 0.07: return "D"
    return "F"

def main():
    print("Loading draft data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    extra_frames = [
        load_extra_draft_file(year, path)
        for year, path in EXTRA_DRAFT_FILES.items()
    ]
    extra_frames = [frame for frame in extra_frames if not frame.empty]
    if extra_frames:
        df = pd.concat([df, *extra_frames], ignore_index=True, sort=False)

    # --- Rename nflverse columns to our standard names ---
    df = df.rename(columns={
        "season":         "year",
        "w_av":           "career_av",
        "games":          "games_played",
        "probowls":       "pro_bowls",
        "allpro":         "all_pro_1st",
        "pfr_player_name":"player",
    })

    # Merge relocated franchises
    df["team"] = df["team"].replace({
    "LVR": "OAK",
    "LAC": "SDG",
    "LAR": "STL",
    })

    # Filter to completed draft classes so career AV has time to mature
    df = df[(df["year"] >= 2010) & (df["year"] <= 2022)].copy()
    print(f"  {len(df)} picks, {df['year'].nunique()} years, {df['team'].nunique()} teams")

    # Clean numerics
    for col in ["career_av", "games_played", "pro_bowls", "all_pro_1st"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["pick"])

    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["pos_group"] = df["position"].apply(map_pos_group)

    # --- Simple ML model: predict career AV ---
    ml_df = df.dropna(subset=["pick", "round", "position", "career_av"]).copy()
    X = pd.get_dummies(ml_df[["pick", "round", "position"]], columns=["position"])
    y = ml_df["career_av"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nRandom Forest R^2 score: {r2_score(y_test, y_pred):.3f}")

    df["predicted_av"] = model.predict(X)
    df["model_delta"] = df["career_av"] - df["predicted_av"]
    df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    # --- Fit expected AV curve ---
    print("\nFitting expected AV curve...")
    expected_curve = fit_expected_curve(df)

    curve_df = expected_curve.reset_index()
    curve_df.columns = ["pick", "expected_av"]
    curve_df.to_csv(CURVE_OUT, index=False)

    # --- Attach expected AV and delta ---
    df["expected_av"] = df["pick"].map(expected_curve)
    df["av_delta"] = df["career_av"] - df["expected_av"]

    top_steals = (
        df.sort_values("av_delta", ascending=False)
        [["player", "team", "pick", "av_delta"]]
        .head(20)
        .copy()
    )
    top_steals["pick"] = top_steals["pick"].astype(int)
    top_steals["av_delta"] = top_steals["av_delta"].round(1)

    # --- Team aggregation ---
    print("\nCalculating team efficiency scores...")
    team_stats = []

    for team, grp in df.groupby("team"):
        best  = grp.loc[grp["av_delta"].idxmax()]
        worst = grp.loc[grp["av_delta"].idxmin()]

        team_stats.append({
            "team":               team,
            "n_picks":            len(grp),
            "total_actual_av":    round(grp["career_av"].sum(), 1),
            "total_expected_av":  round(grp["expected_av"].sum(), 1),
            "total_delta":        round(grp["av_delta"].sum(), 1),
            "avg_delta":          round(grp["av_delta"].mean(), 2),
            "hit_rate":           round((grp["av_delta"] >= 0).mean(), 3),
            "pro_bowlers":        int(grp["pro_bowls"].sum()),
            "all_pros":           int(grp["all_pro_1st"].sum()),
            "best_pick_player":   best["player"],
            "best_pick_round":    int(best["round"]) if not pd.isna(best["round"]) else None,
            "best_pick_av_delta": round(best["av_delta"], 1),
            "worst_pick_player":  worst["player"],
            "worst_pick_round":   int(worst["round"]) if not pd.isna(worst["round"]) else None,
            "worst_pick_av_delta":round(worst["av_delta"], 1),
        })

    team_df = pd.DataFrame(team_stats)
    team_df["grade"] = team_df["avg_delta"].apply(
        lambda s: grade_score(s, team_df["avg_delta"])
    )
    team_df = team_df.sort_values("avg_delta", ascending=False).reset_index(drop=True)
    team_df["rank"] = team_df.index + 1

    team_df.to_csv(TEAM_OUT, index=False)
    team_df.to_json(TEAM_JSON, orient="records", indent=2)
    df.to_csv(PICKS_OUT, index=False)
    top_steals.to_csv(STEALS_OUT, index=False)

    print(f"\n  Saved {TEAM_OUT}, {TEAM_JSON}, {CURVE_OUT}, {PICKS_OUT}, {STEALS_OUT}")

    print("\n=== TOP 10 DRAFTING TEAMS (2010-2022) ===")
    print(team_df[["rank","team","avg_delta","hit_rate","grade"]].head(10).to_string(index=False))

    print("\n=== BOTTOM 5 ===")
    print(team_df[["rank","team","avg_delta","hit_rate","grade"]].tail(5).to_string(index=False))

    print("\n=== CURVE SAMPLE ===")
    for p in [1, 5, 10, 32, 64, 100, 200]:
        if p in expected_curve.index:
            print(f"  Pick {p:3d}: expected AV = {expected_curve[p]:.1f}")

if __name__ == "__main__":
    main()
