"""Streamlit frontend for the Health Condition Prediction System."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend import CATEGORY_ENCODINGS, predict_health


NUMERIC_FIELDS = (
    ("sleep_duration", "Sleep duration", "hours per day", "Example: 7.5", 0.5),
    ("heart_rate", "Heart rate", "bpm", "Example: 72", 1.0),
    ("bmi", "Body mass index", "BMI", "Example: 22.5", 0.1),
    ("calorie_expenditure", "Calorie expenditure", "kcal per day", "Example: 2200", 1.0),
    ("step_count", "Daily step count", "steps", "Example: 8000", 1.0),
    ("exercise_duration", "Exercise duration", "minutes per day", "Example: 45", 1.0),
    ("water_intake", "Water intake", "litres per day", "Example: 2.5", 0.1),
)

CATEGORICAL_FIELDS = (
    ("diet_type", "Diet type", ("Balanced", "Non-Veg", "Veg")),
    ("stress_level", "Stress level", ("Low", "Medium", "High")),
    ("sleep_quality", "Sleep quality", ("Good", "Average", "Poor")),
    ("physical_activity_level", "Physical activity level", ("Active", "Moderate", "Sedentary")),
    ("smoking_alcohol", "Smoking / alcohol", ("No", "Occasional", "Yes")),
    ("gender", "Gender", ("Female", "Male", "Other")),
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Health Condition Prediction System",
        page_icon=":heart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        #MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        :root {
            --ink: #102a43;
            --muted: #5d7285;
            --line: #dbe8ef;
            --blue: #1769aa;
            --green: #15866d;
            --amber: #aa640f;
            --red: #a33434;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            color: var(--ink);
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                linear-gradient(135deg, rgba(255,255,255,.92), rgba(242,249,251,.86)),
                radial-gradient(circle at 0% 8%, rgba(23,105,170,.13), transparent 27rem),
                radial-gradient(circle at 100% 2%, rgba(21,134,109,.12), transparent 30rem),
                #f6fafb;
        }

        .block-container {
            max-width: 1220px;
            padding: 1.5rem 1.25rem 2.4rem;
        }

        .hero {
            min-height: 230px;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(270px, .65fr);
            gap: 1.25rem;
            align-items: stretch;
            margin-bottom: 1rem;
        }

        .hero-main, .stat-panel, [data-testid="stForm"] {
            border: 1px solid rgba(255,255,255,.9);
            background: rgba(255,255,255,.82);
            box-shadow: 0 18px 48px rgba(26, 69, 103, .10);
            backdrop-filter: blur(16px);
        }

        .hero-main {
            border-radius: 8px;
            padding: clamp(1.35rem, 3vw, 2.3rem);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .eyebrow {
            color: var(--green);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }

        .hero h1 {
            max-width: 760px;
            color: #0f3758;
            font-size: clamp(2rem, 5vw, 4.15rem);
            line-height: 1.02;
            letter-spacing: 0;
            margin: 0;
        }

        .hero p {
            max-width: 680px;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.7;
            margin: 1rem 0 0;
        }

        .stat-panel {
            border-radius: 8px;
            padding: 1.2rem;
            display: grid;
            align-content: center;
            gap: .8rem;
        }

        .stat {
            border-left: 3px solid var(--blue);
            padding: .7rem .85rem;
            background: #f8fbfd;
        }

        .stat strong {
            display: block;
            color: #153d5c;
            font-size: 1.05rem;
            margin-bottom: .15rem;
        }

        .stat span {
            color: var(--muted);
            font-size: .88rem;
        }

        .section-title {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: end;
            margin: 1.2rem 0 .8rem;
        }

        .section-title h2 {
            color: #0f3758;
            font-size: 1.35rem;
            margin: 0 0 .25rem;
        }

        .section-title p, .helper {
            color: var(--muted);
            margin: 0;
        }

        .helper {
            font-size: .92rem;
            text-align: right;
        }

        [data-testid="stForm"] {
            border-radius: 8px;
            padding: 1.25rem 1.35rem 1.45rem;
        }

        [data-testid="stForm"] h3 {
            color: #153d5c;
            font-size: 1rem;
            margin: .8rem 0 .75rem;
            padding-bottom: .55rem;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label {
            color: #274d68;
            font-weight: 700;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            min-height: 45px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfdff;
            transition: border-color .18s ease, box-shadow .18s ease;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 45px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfdff;
        }

        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus {
            border-color: #2477b9;
            box-shadow: 0 0 0 3px rgba(36, 119, 185, .12);
        }

        [data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 2.9rem;
            border-radius: 8px;
            font-weight: 800;
            transition: transform .16s ease, box-shadow .16s ease;
        }

        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            border: 0;
            color: #fff;
            background: linear-gradient(120deg, #1769aa, #15866d);
            box-shadow: 0 10px 20px rgba(23,105,170,.18);
        }

        [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
        }

        .result {
            margin-top: 1rem;
            border-radius: 8px;
            border: 1px solid;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 16px 35px rgba(26, 69, 103, .10);
        }

        .result h3 {
            margin: 0 0 .35rem;
            font-size: 1.25rem;
        }

        .result p {
            margin: 0;
            color: inherit;
            line-height: 1.65;
        }

        .fit {
            color: #0f5d4d;
            border-color: #a9dccf;
            background: #eefaf7;
        }

        .risk {
            color: var(--amber);
            border-color: #efd49d;
            background: #fff8e8;
        }

        .unhealthy {
            color: var(--red);
            border-color: #efb8b8;
            background: #fff2f2;
        }

        .page-footer {
            color: var(--muted);
            font-size: .84rem;
            line-height: 1.65;
            margin-top: 1.15rem;
            text-align: center;
        }

        @media (max-width: 840px) {
            .block-container { padding: 1rem .75rem 1.8rem; }
            .hero { grid-template-columns: 1fr; }
            .section-title { display: block; }
            .helper { text-align: left; margin-top: .25rem; }
            [data-testid="stForm"] { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_form() -> None:
    for key, *_ in (*NUMERIC_FIELDS, *CATEGORICAL_FIELDS):
        st.session_state.pop(key, None)
    st.session_state.pop("prediction_result", None)


def numeric_inputs() -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    columns = st.columns(2, gap="large")
    for index, (key, label, unit, placeholder, step) in enumerate(NUMERIC_FIELDS):
        with columns[index % 2]:
            values[key] = st.number_input(
                f"{label} ({unit})",
                value=None,
                min_value=0.0,
                step=step,
                placeholder=placeholder,
                key=key,
            )
    return values


def categorical_inputs() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    columns = st.columns(2, gap="large")
    for index, (key, label, options) in enumerate(CATEGORICAL_FIELDS):
        with columns[index % 2]:
            values[key] = st.selectbox(
                label,
                options,
                index=None,
                placeholder=f"Select {label.lower()}",
                key=key,
            )
    return values


def _display_values(key: str) -> list[str]:
    for field_key, _label, options in CATEGORICAL_FIELDS:
        if field_key == key:
            return list(options)
    return [value.title() for value in CATEGORY_ENCODINGS[key] if value != "unknown"]


def validate(numeric: dict[str, float | None], categorical: dict[str, str | None]) -> list[str]:
    errors: list[str] = []

    if any(value is None for value in numeric.values()):
        errors.append("Complete every numerical health metric.")
    elif any(value <= 0 for value in numeric.values() if value is not None):
        errors.append("All numerical health metrics must be positive.")

    heart_rate = numeric.get("heart_rate")
    bmi = numeric.get("bmi")
    water = numeric.get("water_intake")
    sleep = numeric.get("sleep_duration")
    exercise = numeric.get("exercise_duration")

    if heart_rate is not None and not 30 <= heart_rate <= 220:
        errors.append("Heart rate must be between 30 and 220 bpm.")
    if bmi is not None and not 10 <= bmi <= 60:
        errors.append("BMI must be between 10 and 60.")
    if water is not None and water > 20:
        errors.append("Water intake should be 20 litres or less.")
    if sleep is not None and sleep > 24:
        errors.append("Sleep duration cannot be more than 24 hours.")
    if exercise is not None and exercise > 1440:
        errors.append("Exercise duration cannot be more than 1440 minutes.")

    for key, value in categorical.items():
        normalized = str(value or "").strip().casefold()
        if not normalized:
            errors.append(f"Complete {key.replace('_', ' ')}.")
        elif normalized not in CATEGORY_ENCODINGS[key]:
            allowed = ", ".join(_display_values(key))
            errors.append(f"{key.replace('_', ' ').title()} must be one of: {allowed}.")

    return errors


def result_card(prediction: str) -> None:
    cards = {
        "fit": (
            "fit",
            "Fit",
            "Your indicators currently look healthy. Keep the same consistent routine.",
        ),
        "at-risk": (
            "risk",
            "At-risk",
            "Some indicators need attention. Improve your routine and monitor your health regularly.",
        ),
        "unhealthy": (
            "unhealthy",
            "Unhealthy",
            "Your indicators suggest a health concern. Please consider consulting a qualified doctor.",
        ),
    }
    if prediction not in cards:
        st.error("The prediction service returned an unsupported result.")
        return

    css_class, title, message = cards[prediction]
    st.markdown(
        f'<section class="result {css_class}" role="status"><h3>{title}</h3><p>{message}</p></section>',
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()

    st.markdown(
        """
        <section class="hero">
            <div class="hero-main">
                <div class="eyebrow">Machine learning health assessment</div>
                <h1>Health Condition Prediction System</h1>
                <p>Enter everyday health, activity, and lifestyle details to estimate whether the current condition is fit, at-risk, or unhealthy.</p>
            </div>
            <aside class="stat-panel" aria-label="Assessment details">
                <div class="stat"><strong>13 inputs</strong><span>Vitals, activity, hydration, diet, sleep, and lifestyle.</span></div>
                <div class="stat"><strong>Private session</strong><span>Values are used only to create this prediction.</span></div>
                <div class="stat"><strong>Model ready</strong><span>Uses the trained local model and scaler files.</span></div>
            </aside>
        </section>
        <div class="section-title">
            <div>
                <h2>Health Assessment</h2>
                <p>Fill every field before generating a prediction.</p>
            </div>
            <p class="helper">Select values such as Balanced, Medium, Average, Active, No, and Male.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("health_form", clear_on_submit=False):
        st.markdown("### Vital and Activity Metrics")
        numbers = numeric_inputs()

        st.markdown("### Lifestyle Information")
        categories = categorical_inputs()

        st.markdown("<br>", unsafe_allow_html=True)
        predict_column, reset_column = st.columns([2, 1], gap="medium")
        with predict_column:
            submitted = st.form_submit_button(
                "Predict Health Condition",
                type="primary",
                use_container_width=True,
            )
        with reset_column:
            st.form_submit_button("Reset Form", use_container_width=True, on_click=reset_form)

    if submitted:
        st.session_state.pop("prediction_result", None)
        errors = validate(numbers, categories)
        if errors:
            st.error("\n\n".join(f"- {error}" for error in errors))
        else:
            input_data: dict[str, Any] = {**numbers, **categories}
            try:
                with st.spinner("Analyzing health information..."):
                    prediction = str(predict_health(input_data)).strip().lower()
                st.session_state.prediction_result = prediction
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")

    if prediction := st.session_state.get("prediction_result"):
        result_card(prediction)

    st.markdown(
        """
        <footer class="page-footer">
            <strong>Health Condition Prediction System</strong><br>
            This tool supports quick screening only and does not replace professional medical advice.
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
