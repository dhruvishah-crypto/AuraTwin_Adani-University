# RespiraTwin Data Dictionary

All fields are categorized with explicit data provenance:
- **`sourced`**: public real-world benchmark data (e.g., public air quality stations).
- **`simulated`**: generated via physiological models, stochastic processes, or EHR distributions.
- **`scenario`**: hypothetical what-if parameter variations.

---

## 1. Patient EHR Profiles (`data/patients.csv`)

| Field Name | Type | Range / Allowed Values | Provenance | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | string | `PT0001` - `PTNNNN` | simulated | Unique identifier for synthetic virtual patient. |
| `age` | int | 5 - 85 | simulated | Patient age in years. |
| `sex` | string | `M`, `F` | simulated | Biological sex. |
| `asthma_severity` | string | `Mild`, `Moderate`, `Severe` | simulated | Clinician-assessed baseline asthma severity. |
| `prior_exacerbations_12m` | int | 0 - 15 | simulated | Number of exacerbation episodes in previous 12 months. |
| `controller_prescribed` | int | 0 or 1 | simulated | Whether daily controller (e.g. ICS/LABA) is prescribed. |
| `baseline_resp_rate` | float | 10.0 - 32.0 | simulated | Baseline resting respiratory rate (breaths/min). |
| `baseline_spo2` | float | 88.0 - 100.0 | simulated | Baseline resting oxygen saturation (%). |
| `comorbidities` | string | Comma-separated | simulated | Diagnosed comorbidities (`allergic_rhinitis`, `GERD`, `obesity`, `hypertension`, `none`). |
| `pollution_sensitivity` | string | `Low`, `Medium`, `High` | simulated | Sensitivity phenotype to airborne pollutants. |

> **Isolated Ground Truth (`data/latent_truth.csv`)**:
> - `latent_vulnerability`: continuous in `[0.0, 1.0]`, provenance `simulated (HIDDEN)`.
> - Underlying biological factor strictly for the event simulator. **NEVER** used in model feature pipelines.

---

## 2. Ambient Air Quality & Pollution (`data/sample_pollution.csv`)

> **CRITICAL LABELING & MODELING NOTICE**:
> - **Primary Model Features**: Prefer physical concentrations **`pm25`** and **`pm10`** ($\mu g/m^3$) as feature inputs for all machine learning models.
> - **Derived Index Labeling**: The `aqi` column is specifically a **"Derived US-EPA-style AQI (from PM2.5)"**. It is **NOT** India's CPCB National Air Quality Index (NAQI). It must never be referred to simply as "AQI" without this qualifier in dashboards or documentation, and raw PM2.5 concentration must never be called "AQI".
> - **Current Data Mode**: The bundled sample file contains **synthetic fallback data calibrated to CPCB Delhi ITO station statistics** (enabling full offline reproducibility).

| Field Name | Type | Units / Allowed Values | Provenance | Description & Labeling Requirement |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | string | ISO datetime | simulated / sourced | Hourly observation timestamp (`YYYY-MM-DD HH:MM:SS`). |
| `pm25` | float | $\mu g/m^3$ ($0 - 500+$) | simulated / sourced | **Primary Feature**: Particulate matter $< 2.5\,\mu m$. Never call this AQI. |
| `pm10` | float | $\mu g/m^3$ ($0 - 600+$) | simulated / sourced | **Primary Feature**: Particulate matter $< 10\,\mu m$. |
| `aqi` | int | Index ($0 - 500$) | simulated (derived) | **Derived US-EPA-style AQI (from PM2.5)**. Computed via US EPA breakpoint formulas. NOT India's CPCB index. Use only as secondary context, not primary model feature. |
| `city` | string | e.g., `Delhi`, `London` | simulated / sourced | Reference urban geography. |
| `source` | string | e.g., `sourced_sample` / `synthetic` | simulated | Data provenance tag indicating real physical sensor vs synthetic fallback. |
| `last_updated`| string | ISO timestamp | metadata | Timestamp when dataset was generated or ingested. |

---

## 3. Wearable & Symptom Time Series (`data/patient_timeseries.csv`)

| Field Name | Type | Units / Range | Provenance | Description |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | string | `PT0001` - `PTNNNN` | simulated | Patient identifier linked to EHR profile. |
| `timestamp` | string | ISO datetime | simulated | Hourly timestamp aligned with environmental series. |
| `respiratory_rate` | float | breaths/min ($10 - 45$) | simulated | Hourly resting respiratory rate. Contains realistic dropouts (NaN). |
| `heart_rate` | float | bpm ($45 - 160$) | simulated | Continuous heart rate with circadian rhythm. Contains dropouts (NaN). |
| `hrv_rmssd_ms` | float | milliseconds ($8 - 110$) | simulated | Heart rate variability (RMSSD). Inversely shifts with distress. |
| `spo2` | float | % ($75 - 100$) | simulated | Blood oxygen saturation with nocturnal and distress dips. Contains NaNs. |
| `sleep_hours` | float | hours ($4.0 - 10.0$) | simulated | Daily sleep duration summary. |
| `steps` | int | step count ($0 - 3000$) | simulated | Hourly step activity. Near zero during sleep hours. |
| `controller_taken` | int | 0 or 1 | simulated | Daily controller medication adherence indicator. |
| `rescue_puffs` | int | 0 - 6 | simulated | SABA rescue inhaler puff actuations; reactively follows symptom distress. |
| `symptom_score` | int | 0 - 10 | simulated | Patient diary symptom severity rating ($0=\text{none}$, $10=\text{acute crisis}$). |
| `provenance` | string | `"simulated"` | simulated | Data origin tag. |
