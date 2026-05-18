from __future__ import annotations

import json

import requests
import streamlit as st

st.set_page_config(page_title="WHO Health Predictor", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #1f2937, #0b1020 60%);
            color: #f9fafb;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #cbd5e1;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">WHO Health Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">BentoML inference endpoint UI with JSON input.</div>',
    unsafe_allow_html=True,
)

endpoint = st.text_input("BentoML predict URL", "http://127.0.0.1:3000/predict")
sample = {
    "population": 1_400_000_000,
    "population_growth": 0.9,
    "urban_pct": 36.0,
    "death_rate": 7.3,
    "birth_rate": 17.4,
    "health_expenditure_pct_gdp": 3.0,
    "health_expenditure_per_capita": 73.0,
    "physicians_per_1000": 0.9,
    "hospital_beds_per_1000": 0.5,
    "nurses_per_1000": 1.7,
    "under5_mortality": 34.0,
    "neonatal_mortality": 22.0,
    "maternal_mortality": 103.0,
    "tb_incidence": 188.0,
    "hiv_incidence": 0.04,
    "communicable_death_pct": 26.0,
    "noncommunicable_death_pct": 63.0,
    "smoking_male": 21.0,
    "smoking_female": 2.0,
    "obesity_pct": 5.0,
    "alcohol_per_capita": 5.6,
    "diabetes_pct": 8.9,
    "safe_water_pct": 93.0,
    "sanitation_pct": 62.0,
    "handwashing_pct": 59.0,
    "immunization_measles": 89.0,
    "immunization_dpt": 91.0,
    "immunization_bcg": 92.0,
    "immunization_pol3": 90.0,
    "immunization_hib3": 89.0,
    "stunting_pct": 35.0,
    "wasting_pct": 19.0,
    "overweight_children_pct": 2.0,
    "undernourishment_pct": 14.0,
    "fertility_rate": 2.0,
    "adolescent_fertility": 15.0,
    "prenatal_care_pct": 84.0,
    "gdp_per_capita": 2500.0,
    "gdp_per_capita_ppp": 8500.0,
    "poverty_rate": 18.0,
    "region": "South Asia",
    "income_level": "Lower middle income",
}

payload_text = st.text_area("Input JSON for prediction", value=json.dumps(sample, indent=2), height=420)

if st.button("Predict", use_container_width=True):
    try:
        payload = json.loads(payload_text)
        response = requests.post(endpoint, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.success("Prediction successful")
        st.json(result)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")

