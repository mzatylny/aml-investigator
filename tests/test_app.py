from pathlib import Path
from streamlit.testing.v1 import AppTest
import aml.data
import aml.model
import numpy as np
import streamlit as st


def test_dashboard_and_threshold_interaction(monkeypatch):
    st.cache_resource.clear()
    generator = aml.data.generate_transactions
    monkeypatch.setattr(aml.data, "generate_transactions", lambda seed: generator(seed=seed, days=12, accounts=50, daily_transactions=30))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert app.title[0].value == "Follow the money."
    app.slider[0].set_value(1.0).run()
    assert not app.exception
    assert app.metric[1].value == "0"
    app.checkbox(key="show_truth").check().run()
    assert not app.exception
    assert app.radio(key="review_model").value == "GraphSAGE"
    app.radio(key="review_model").set_value("Random forest").run()
    assert not app.exception
    app.checkbox(key="include_gnn").uncheck().run()
    assert not app.exception


def test_dashboard_preserves_no_alert_policy_for_saturated_scores(monkeypatch):
    st.cache_resource.clear()
    generator = aml.data.generate_transactions
    run = aml.model.run_experiment
    monkeypatch.setattr(aml.data, "generate_transactions", lambda seed: generator(seed=seed, days=10, accounts=30, daily_transactions=20))

    def saturated(*args, **kwargs):
        experiment = run(*args, **kwargs, gnn_epochs=1)
        experiment.transactions["gnn_score"] = np.float32(1.)
        experiment.gnn.threshold = float(np.nextafter(np.float32(1), np.float32(np.inf)))
        return experiment

    monkeypatch.setattr(aml.model, "run_experiment", saturated)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert app.slider[0].value == 1.01
    assert app.metric[1].value == "0"
    app.slider[0].set_value(1.).run()
    assert not app.exception
    assert app.metric[1].value == app.metric[0].value
    st.cache_resource.clear()
