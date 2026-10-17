# RespiraTwin Architecture

## System Diagram
```
+-------------------------------------------------------------------+
|                           RespiraTwin                             |
+-------------------------------------------------------------------+
      |
      +---> [Data Generation / Ingestion]
      |        |-- Synthetic EHR (patients.csv)
      |        |-- Hidden Latent Truth (data/latent_truth.csv)
      |        |-- Pollution Signals (sample_pollution.csv / OpenAQ)
      |        `-- Simulated Wearables (sample_wearables.csv)
      |
      +---> [Data Validation (load_ehr.py)]
      |        `-- Strict schema, type, and range validation
      |
      +---> [Feature Engineering (build_features.py)]
      |        `-- Point-in-time temporal fusion (no future leakage)
      |
      +---> [Simulator & ML Training]
      |        |-- Event Simulator (uses latent_truth.csv + features)
      |        `-- Scikit-Learn Model (uses features only, patient split)
      |
      `---> [Interactive Dashboard & Digital Twin (app.py)]
               |-- Patient 24-hr Risk Forecast
               |-- Explainable Attributions (SHAP / Tree-based)
               `-- Simulated What-If Scenarios
```

## Anti-Leakage & Safety Guardrails
1. **Latent Truth Isolation**: `latent_vulnerability` is stored exclusively in `data/latent_truth.csv` and never enters `patients.csv` or `build_features.py`.
2. **Clinical Disclaimers**: All dashboard views and metrics explicitly state they measure learning of a simulated process, not real-world prediction.
3. **Reproducibility**: Generators produce seed metadata sidecar JSON files.
