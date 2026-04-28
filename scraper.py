"""
NFL Draft Scraper — Pro Football Reference
Pulls draft data for 2010–2022, saves to data/draft_data.csv

Usage:
    pip install requests beautifulsoup4 pandas
    python scraper.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

YEARS = range(2010, 2023)  # 2010–2022 inclusive
BASE_URL = "https://www.pro-football-reference.com/years/{year}/draft.htm"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "draft_data.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Columns we care about from PFR's draft table
# PFR column headers (th data-stat values):
#   pick_round, pick_number (overall), team, player, pos, college_univ,
#   career_av (career approximate value), w_av (weighted AV), games, etc.
COLS_WANTED = {
    "pick_round":    "round",
    "pick_number":   "pick",        # overall pick number
    "team":          "team",
    "player":        "player",
    "pos":           "position",
    "college_univ":  "college",
    "career_av":     "career_av",   # career Approximate Value — our main metric
    "w_av":          "weighted_av", # weighted AV (discounts old seasons)
    "games":         "games_played",
    "all_pros_first_team": "all_pro_1st",
    "pro_bowls":     "pro_bowls",
}


def scrape_year(year: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    print(f"  Fetching {year}...", end=" ")

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "drafts"})
    if table is None:
        print("table not found, skipping.")
        return pd.DataFrame()

    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        # Skip header rows that PFR repeats mid-table
        if tr.get("class") and "thead" in tr.get("class"):
            continue

        row = {"year": year}
        for stat, col_name in COLS_WANTED.items():
            td = tr.find(["td", "th"], {"data-stat": stat})
            if td:
                # Get inner text; for player links grab the text
                row[col_name] = td.get_text(strip=True)
            else:
                row[col_name] = None

        # Skip rows with no player name (empty/spacer rows)
        if not row.get("player"):
            continue

        rows.append(row)

    print(f"{len(rows)} picks.")
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric columns
    numeric_cols = ["round", "pick", "career_av", "weighted_av",
                    "games_played", "all_pro_1st", "pro_bowls"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no pick number (shouldn't happen but just in case)
    df = df.dropna(subset=["pick"])

    # Fill AV NaN with 0 (undrafted/never played → 0 career value)
    df["career_av"] = df["career_av"].fillna(0)
    df["weighted_av"] = df["weighted_av"].fillna(0)
    df["games_played"] = df["games_played"].fillna(0)
    df["pro_bowls"] = df["pro_bowls"].fillna(0)
    df["all_pro_1st"] = df["all_pro_1st"].fillna(0)

    # Normalize team abbreviations (PFR uses 3-letter codes already)
    df["team"] = df["team"].str.upper().str.strip()

    # Clean position — group rare/blank positions
    df["position"] = df["position"].str.upper().str.strip()
    df["position"] = df["position"].replace({"":"UNK", None:"UNK"})

    return df.reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_frames = []
    for year in YEARS:
        try:
            df_year = scrape_year(year)
            if not df_year.empty:
                all_frames.append(df_year)
        except Exception as e:
            print(f"  ERROR on {year}: {e}")

        # Be polite to PFR — 4 second delay between requests
        time.sleep(4)

    if not all_frames:
        print("No data collected. Exiting.")
        return

    df = pd.concat(all_frames, ignore_index=True)
    df = clean(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df)} total picks saved to {OUTPUT_FILE}")
    print(df.head(10).to_string())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Teams: {df['team'].nunique()} unique")


if __name__ == "__main__":
    main()