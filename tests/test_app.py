from pathlib import Path
from streamlit.testing.v1 import AppTest
import aml.data


def test_dashboard_and_threshold_interaction(monkeypatch):
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
