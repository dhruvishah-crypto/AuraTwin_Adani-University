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
> All risk coefficients, hazard equations, and episode dynamics listed below represent **explicit simulation assumptions** designed to evaluate machine learning learning curves on a synthetic stochastic process. They are **NOT** clinical facts, validated medical guidelines, or real-world predictive risk calculators.

The target label **`event_next_24h`** is generated via an **episode-based exacerbation simulator**:
$$\text{event\_next\_24h}(t) = 1 \iff \exists t_{\text{onset}} \in (t, t + 24\text{h}]$$

### 1. Exogenous Hourly Hazard Model
Discrete exacerbation onsets are sampled sequentially per patient from an hourly hazard driven **strictly by exogenous drivers** (physiological signals are consequences of the episode, never drivers of onset):
$$\text{logit\_hazard}(t) = \beta_0 + \beta_{\text{vuln}} (\text{vuln} - 0.5) + \beta_{\text{sev}} (\text{sev} - 2) + \beta_{\text{prior}} (\min(\text{prior}, 4) - 1.5) + \beta_{\text{pm25}} \max(0, \text{pm25}_{24h} - 35) (1 + 0.35 \cdot \text{poll\_sens}) + \beta_{\text{lapse}} \cdot \text{lapse} + \epsilon_t + \eta_t$$

| Parameter / Factor | Value | Simulator Modeling Assumption (NOT Clinical Fact) |
| :--- | :--- | :--- |
| **Baseline Intercept ($\beta_0$)** | `-5.85` | Calibrates inter-episode frequency to target ~1 episode per 10–25 patient-days (~4.7 episodes / 60 days). |
| **Hidden Latent Vulnerability ($\beta_{\text{vuln}}$)** | `+0.65` | Ground-truth biological susceptibility known only to the simulator (mean 0.5, range [0, 1]). |
| **Asthma Severity ($\beta_{\text{sev}}$)** | `+0.35` | EHR disease phenotype (Mild=1, Moderate=2, Severe=3) modulates baseline hazard. |
| **Prior Exacerbations ($\beta_{\text{prior}}$)** | `+0.12` | Historical 12-month exacerbation count reflects personal recurrence phenotype. |
| **Ambient PM2.5 Exposure ($\beta_{\text{pm25}}$)** | `+0.012` | Rolling 24-hour PM2.5 exposure exceeding $35\,\mu\text{g/m}^3$, scaled by individual pollution sensitivity (Low=1, Med=2, High=3). |
| **Controller Adherence Lapse ($\beta_{\text{lapse}}$)** | `+0.55` | Exogenous missed daily controller medication intake adds airway vulnerability on lapse days. |
| **Gaussian Noise ($\epsilon_t$)** | `\mathcal{N}(0, 0.35^2)` | Uncorrelated hourly physiological variability. |
| **Stochastic Shocks ($\eta_t$)** | `P=1.2%, [1.0, 2.0]` | Models unmeasured acute triggers (e.g., viral exposure, allergen spike, physical overexertion). |

### 2. Episode Dynamics & Prodromal Window
When an onset triggers at hour $t_{\text{onset}}$:
1. **Refractory Period**: A **72-hour refractory period** is enforced after each episode ends during which hazard is strictly zero.
2. **Episode Duration**: Uniformly sampled $D \in [12, 48]$ hours.
3. **Prodromal Window**: Uniformly sampled $W \in [6, 24]$ hours prior to onset ($[t_{\text{onset}} - W, t_{\text{onset}})$).
4. **Silent Episodes**: **30% of episodes** are randomly designated as "silent" (weak or negligible prodromal drift, prodrome multiplier $\approx 0.08$), preventing trivial detection.
5. **Non-Silent Prodromal Drift**: Accelerating non-linear ramp $\phi(t) = \tau^{1.3} \cdot \alpha$:
   - **Respiratory Rate**: Rises $+ (2.0 - 4.5)\,\text{bpm}$ above baseline.
   - **SpO2**: Drops $- (1.5 - 3.5)\%$ below baseline.
   - **Rescue Inhaler Puffs**: Escalating puff frequency (1–3 puffs).
   - **Symptom Diary**: Worsens by $+ (2.0 - 4.0)$ points on 0–10 scale.
   - **Heart Rate / HRV**: Mild tachycardia and vagal withdrawal.
6. **Acute Flare**: Signals remain severely elevated throughout $[t_{\text{onset}}, t_{\text{onset}} + D)$ (RR $+5.5\,\text{bpm}$, SpO2 $-4.0\%$, symptom score 6–9, rescue puffs 3–6).
7. **Recovery Phase**: 18-hour smooth decay returning physiological channels to baseline.
8. **Evaluation Storage**: Onset timestamps and episode attributes are saved to `data/episodes.csv` strictly for evaluation and alarm-fatigue modeling. Onset flags and episode metadata are never used as model features.
