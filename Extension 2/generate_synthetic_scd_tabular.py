"""Generate synthetic tabular data for sickle-cell comorbidity risk modeling.

Creates a CSV with demographic, lab, vital-sign, and binary comorbidity
columns along with synthetic stroke and heart_failure labels.  The image
probability column (scd_prob_image) is also synthesized.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import SYNTHETIC_DIR


SEED = 42
N_ROWS = 500


def _make_row(rng: np.random.Generator):
    sex = rng.choice(["F", "M"])
    age = int(rng.integers(5, 75))
    hb = round(float(rng.normal(9.5 if sex == "F" else 10.5, 1.8)), 1)
    hb = max(5.0, min(20.0, hb))
    wbc = round(float(rng.normal(10.0, 3.5)), 1)
    wbc = max(2.0, min(30.0, wbc))
    platelets = int(rng.normal(320, 90))
    platelets = max(50, min(1000, platelets))
    spo2 = round(float(rng.normal(95.5, 2.5)), 1)
    spo2 = max(80.0, min(100.0, spo2))
    sbp = int(rng.normal(118, 14))
    sbp = max(70, min(240, sbp))
    dbp = int(rng.normal(72, 10))
    dbp = max(30, min(140, dbp))
    smoker = int(rng.random() < 0.18)
    diabetes = int(rng.random() < 0.12)
    hypertension = int(rng.random() < 0.25)
    pulm_htn = int(rng.random() < 0.10)

    # Synthetic image prob
    scd_prob = round(float(rng.beta(2.5, 1.5)), 4)

    # Synthetic outcomes (loosely correlated)
    stroke_logit = (
        -3.5
        + 0.03 * age
        + 1.2 * scd_prob
        + 0.5 * hypertension
        + 0.3 * smoker
        - 0.1 * (hb - 10.0)
        + 0.2 * pulm_htn
    )
    stroke = int(rng.random() < (1.0 / (1.0 + np.exp(-stroke_logit))))

    hf_logit = (
        -4.0
        + 0.025 * age
        + 1.0 * scd_prob
        + 0.4 * pulm_htn
        + 0.3 * diabetes
        - 0.15 * (hb - 10.0)
        + 0.2 * hypertension
    )
    heart_failure = int(rng.random() < (1.0 / (1.0 + np.exp(-hf_logit))))

    return {
        "sex": sex,
        "age_years": age,
        "hemoglobin_g_dl": hb,
        "wbc_10e9_l": wbc,
        "platelets_10e9_l": platelets,
        "spo2_percent": spo2,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "smoker": smoker,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "pulmonary_hypertension": pulm_htn,
        "scd_prob_image": scd_prob,
        "stroke": stroke,
        "heart_failure": heart_failure,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SCD tabular data")
    parser.add_argument("--n", type=int, default=N_ROWS, help="Number of rows")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(SYNTHETIC_DIR / "synthetic_scd_heart_stroke.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    rows = [_make_row(rng) for _ in range(args.n)]
    df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Stroke prevalence:  {df['stroke'].mean():.2%}")
    print(f"HF prevalence:      {df['heart_failure'].mean():.2%}")


if __name__ == "__main__":
    main()
