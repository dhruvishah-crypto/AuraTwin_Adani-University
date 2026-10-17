You are building "RespiraTwin", a proof-of-concept healthcare Digital Twin for the Happiest Health Digital Twin Challenge 2026. Build it in stages. Do not skip ahead. After each stage, run what you built and report what works and what is broken.

GOAL
A Python app that fuses (1) static synthetic EHR profiles and (2) dynamic environmental + simulated wearable time series to estimate the probability of a SIMULATED asthma exacerbation-risk event within the next 24 hours for a virtual patient, with a doctor-facing Streamlit dashboard and a what-if simulator.

HARD RULES
- Only synthetic or open data. No real patient data.
- Event labels come from a simulator. Never claim clinical accuracy. Every README, dashboard page and metric table must say results measure learning of a simulated process, not real-world prediction.
- Never call a PM2.5 concentration "AQI". Keep pm25, pm10 (ug/m3) and aqi (official index, only if sourced) as separate fields.
- Record the provenance of every field: "sourced" (public data), "simulated" (wearable/adherence/symptoms/EHR), or "scenario" (hypothetical what-if input). Show provenance in the UI.
- Reproducibility: every generator takes a seed and writes a config JSON beside its output.
- No data leakage: features at time t use only observations at or before t. Split train/validation/test by patient_id, never by random rows. Fit scalers/imputers only on training patients.
- Keep it simple: Python, pandas, numpy, scikit-learn, Streamlit, Plotly, joblib, pytest. No deep learning, no cloud DB, no React.
- Pollution source order: (1) a bundled attributed sample CSV, (2) optional fetch from OpenAQ or CPCB, (3) synthetic fallback. The app must run fully offline.
- What-if scenarios must never overwrite stored patient data and must not recommend changing medication. Label them "simulated scenario".

REPO STRUCTURE (create exactly this)
respiratwin/
  app.py, requirements.txt, README.md, LICENSE (MIT), .gitignore
  data/ (sample_patients.csv, sample_pollution.csv, sample_wearables.csv)
  src/ generate_data.py, load_ehr.py, ingest_pollution.py, simulate_wearables.py, build_features.py, simulate_events.py, train_model.py, predict_risk.py, digital_twin.py, explain_prediction.py
  models/.gitkeep
  tests/ test_data_validation.py, test_features.py, test_predictions.py
  docs/ data_dictionary.md, architecture.md, demo_screenshots/
  notebooks/exploratory_analysis.ipynb

CODE QUALITY
Type hints, docstrings, small functions, clear errors, no hardcoded absolute paths. Add a Makefile or run_all.sh with: generate -> train -> test -> run app.