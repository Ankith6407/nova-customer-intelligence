# NOVA 2.0 — Advanced Customer Intelligence Platform

## Required folder structure

```text
Customer_Segmentation_Dashboard/
├── app.py
├── requirements.txt
└── customer_segmentation_outputs/
    ├── customer_segments.csv
    ├── segment_profile.csv
    ├── campaign_analysis.csv
    └── cluster_summary.csv
```

## Run locally (Windows / Anaconda)

Open Anaconda Prompt, change directory to this folder, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `python` is not recognised, activate the same Conda environment used by Jupyter first. Alternatively, install the listed packages in that environment through Anaconda.

## What's new
- Dynamic executive signals recalculate with sidebar filters.
- New Growth Playbook tab with segment-specific retention, re-engagement and development hypotheses.
- Downloadable segment action matrix.
- Campaign measurement guidance using holdout groups.
- Existing overview, profiles, campaign analysis, customer explorer and methodology sections retained.

## Important
The dashboard reads the four notebook CSV exports; it does not retrain clustering models. Keep the CSV files inside `customer_segmentation_outputs` beside `app.py`. Segment labels are based on observed average spending. Recommendations are test hypotheses, not forecasts or guaranteed ROI.
