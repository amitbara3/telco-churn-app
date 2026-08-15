"""Tests for the Streamlit UI's actual submit path.

This is the surface that matters most in deployment: Hugging Face Spaces
exposes port 7860, so the UI is the *public* face of the app while the
FastAPI backend stays internal. Until now it was also the least tested
part — CI only checked that the page returned HTTP 200, which a page
throwing an exception on submit would still do.

`AppTest` runs the script headlessly, so widget defaults, payload
construction and result rendering are all exercised for real. The API call
is stubbed: this is testing the UI's half of the contract, and
tests/test_api.py already covers the server's half.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
SCHEMA = json.loads(
    (Path(__file__).resolve().parent.parent / "model" / "feature_schema.json").read_text()
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def run_and_submit(api_payload=None):
    """Render the app, click Predict, and return (AppTest, captured request)."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(
            api_payload
            or {"churn_probability": 0.6804, "churn_prediction": "Yes", "risk_level": "High"}
        )

    at = AppTest.from_file(APP, default_timeout=30)
    with patch("requests.post", side_effect=fake_post):
        at.run()
        assert not at.exception, f"app raised on first render: {at.exception}"
        at.button[0].click().run()
        assert not at.exception, f"app raised on submit: {at.exception}"
    return at, captured


def test_app_renders_without_error():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.button, "no submit button rendered"


def test_submit_sends_every_field_the_api_requires():
    """The form must produce a payload the API accepts — this is exactly
    the train/serve-style contract that breaks silently on a rename."""
    from app.schemas import CustomerFeatures

    _, captured = run_and_submit()
    assert captured, "clicking Predict sent no request"

    # Validates against the real request model: a missing or misnamed
    # field, or an out-of-range default, fails here.
    CustomerFeatures(**captured["json"])


def test_submitted_values_come_from_the_widgets():
    """Guards against a payload built from stale or hardcoded values
    instead of what the user actually selected."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.slider[0].set_value(48)  # tenure
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResponse(
            {"churn_probability": 0.12, "churn_prediction": "No", "risk_level": "Low"}
        )

    with patch("requests.post", side_effect=fake_post):
        at.button[0].click().run()

    assert not at.exception
    assert captured["tenure"] == 48


def test_prediction_result_is_rendered():
    at, _ = run_and_submit(
        {"churn_probability": 0.6804, "churn_prediction": "Yes", "risk_level": "High"}
    )
    shown = [m.value for m in at.metric]
    assert any("68" in str(v) for v in shown), f"probability not displayed: {shown}"
    assert any("Yes" in str(v) for v in shown), f"prediction not displayed: {shown}"


def test_api_failure_is_surfaced_not_swallowed():
    """A backend outage must show an error, not a blank page or a stale
    number the user would read as a real prediction."""
    import requests

    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    with patch("requests.post", side_effect=requests.RequestException("connection refused")):
        at.button[0].click().run()

    assert not at.exception, "UI crashed instead of reporting the failure"
    assert at.error, "no error shown to the user when the API was unreachable"


@pytest.mark.parametrize("column", list(SCHEMA["categorical_features"]))
def test_every_dropdown_offers_only_values_the_api_accepts(column):
    """The UI builds its dropdowns from feature_schema.json while the API
    validates against Literal types in schemas.py. Those are two separate
    sources that must agree."""
    from app.schemas import CustomerFeatures

    field = CustomerFeatures.model_fields[column]
    allowed = set(getattr(field.annotation, "__args__", ()))
    offered = set(SCHEMA["categorical_features"][column])
    assert offered <= allowed, f"{column}: UI offers {offered - allowed}, API rejects them"
