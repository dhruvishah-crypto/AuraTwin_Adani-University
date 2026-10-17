# RespiraTwin Data Dictionary

All fields are categorized with explicit data provenance:
- **`sourced`**: public real-world benchmark data (e.g., public air quality stations).
- **`simulated`**: generated via physiological models, stochastic processes, or EHR distributions.
- **`scenario`**: hypothetical what-if parameter variations.

| Field Name | Type | Range / Allowed Values | Provenance | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | string | `PT001` - `PTNNN` | simulated | Unique identifier for synthetic virtual patient. |
| `age` | int | 5 - 85 | simulated | Patient age in years. |
| `sex` | string | `M`, `F` | simulated | Biological sex. |
| `asthma_severity` | string | `Mild`, `Moderate`, `Severe` | simulated | Clinician-assessed baseline asthma severity. |
| `prior_exacerbations_12m` | int | 0 - 15 | simulated | Number of exacerbation episodes in previous 12 months. |
| `controller_prescribed` | bool / int | 0 or 1 | simulated | Whether daily controller (e.g. ICS/LABA) is prescribed. |
| `baseline_resp_rate` | float | 10.0 - 32.0 | simulated | Baseline resting respiratory rate (breaths/min). |
| `baseline_spo2` | float | 88.0 - 100.0 | simulated | Baseline resting oxygen saturation (%). |
| `comorbidities` | string | Comma-separated | simulated | Diagnosed comorbidities (e.g., `allergic_rhinitis`, `GERD`, `obesity`, `none`). |
| `pollution_sensitivity` | string | `Low`, `Medium`, `High` | simulated | Sensitivity phenotype to airborne pollutants. |
| `latent_vulnerability` | float | 0.0 - 1.0 | simulated (HIDDEN) | Underlying biological vulnerability factor stored exclusively in `data/latent_truth.csv` for the event simulator. NEVER used in feature matrices or models. |
