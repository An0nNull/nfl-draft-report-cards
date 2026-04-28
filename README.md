# NFL Draft Report Cards

A Python and Flask dashboard that grades NFL team draft performance from 2010-2022.

The project compares each player's actual career Approximate Value (AV) to the expected value of their draft slot, then turns that into team report cards, charts, and plain-English explanations.

## Features

- Team draft efficiency rankings
- Draft grades for all 32 franchises
- Best picks and biggest underperformers by team
- Position-level breakdowns
- Expected AV curve by draft slot
- Random Forest model for predicted career AV
- Scatter plot comparing predicted AV to actual career AV
- Simple dashboard built with Flask, pandas, scikit-learn, and Chart.js

## Why 2010-2022?

Career AV is an accumulated stat, so recent draft classes have not had enough time to build a fair career sample. The analysis uses 2010-2022 to avoid unfairly labeling very young players as misses.

## Tech Stack

- Python
- Flask
- pandas
- NumPy
- SciPy
- scikit-learn
- BeautifulSoup
- Chart.js
- HTML/CSS/JavaScript

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the analysis outputs:

```bash
python analysis.py
```

Start the dashboard:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Data Notes

The draft data comes from Pro Football Reference style draft data and local Sports Reference exports. Approximate Value is used as a simple cross-position career value metric.

This is a sports analytics project, not a final scouting model. The results are best read as historical team drafting efficiency, not player-by-player scouting truth.

## Security Notes

The Flask app is configured for local use, with debug mode disabled and only the required dashboard data files exposed.

