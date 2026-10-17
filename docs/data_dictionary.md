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

---

## 4. Simulated Exacerbation Event Modeling & Assumptions (`src/simulate_events.py`)

> **CLINICAL DISCLAIMER**:
> All risk coefficients listed below represent **explicit modeling assumptions** designed to evaluate machine learning learning curves on a simulated stochastic process. They are **NOT** clinical facts, validated medical guidelines, or real-world predictive risk calculators.

The target label **`event_next_24h`** is generated via a transparent logit risk model with stochastic shocks:
$$P(\text{event}_{t+1:t+24} = 1) = \frac{1}{1 + \exp(-z_{i, t})}$$

| Parameter / Factor | Value | Modeling Assumption (NOT Clinical Fact) |
| :--- | :--- | :--- |
| **Intercept ($\beta_0$)** | `-7.40` | Positions cohort baseline daily risk to achieve a realistic demo event rate of ~8-11%. |
| **Hidden Latent Truth ($\beta_{\text{latent}}$)** | `+1.60` | Biological airway vulnerability known only to the simulator. Ensures models cannot achieve deterministic overfit. |
| **Asthma Severity ($\beta_{\text{sev}}$)** | `+0.45` | Structural disease severity phenotype (0=Mild, 1=Moderate, 2=Severe) raises baseline exacerbation susceptibility. |
| **Prior Exacerbations ($\beta_{\text{exac}}$)** | `+0.28` | History of exacerbations in the preceding 12 months reflects individual frequency phenotype. |
| **PM2.5 Exposure ($\beta_{\text{pm25}}$)** | `+0.018` | Prolonged 24h rolling PM2.5 above $35\,\mu g/m^3$ triggers inflammation, amplified by pollution sensitivity. |
| **Missed Controller ($\beta_{\text{adh}}$)** | `+0.85` | Lapsed daily controller medication adherence over the past 7 days increases airway instability. |
| **Rescue Puffs ($\beta_{\text{rescue}}$)** | `+0.32` | Escalating rescue SABA inhaler actuations indicate active symptom breakthrough. |
| **Symptom Score ($\beta_{\text{sym}}$)** | `+0.25` | Patient-reported symptom diary ratings above baseline indicate escalating distress. |
| **Respiratory Rate Deviation ($\beta_{\text{rr}}$)** | `+0.15` | Tachypnea (breaths/min elevated above personal baseline) reflects physiologic compensation. |
| **SpO2 Drop ($\beta_{\text{spo2}}$)** | `+0.35` | Oxygen saturation deficit below personal baseline reflects acute gas-exchange impairment. |
| **Heart Rate Elevation ($\beta_{\text{hr}}$)** | `+0.04` | Acute resting heart rate elevation relative to 72h baseline signals autonomic distress. |
| **Gaussian Noise ($\sigma$)** | `0.45` | Idiosyncratic micro-variations and physiological fluctuation noise. |
| **Unobserved Shock Probability** | `3.5%` | Models unmeasured real-world triggers (viral infections, indoor irritants, emotional distress). |
| **Shock Magnitude** | `[1.2, 2.5]` | Additive logit boost when an unobserved shock occurs. |
