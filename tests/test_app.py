from streamlit.testing.v1 import AppTest

from churn_app.config import PROJECT_ROOT


def test_app_starts_with_demo_data():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Telecom churn modelling workbench"
    assert app.tabs[0].label == "Model evidence"
    assert len(app.metric) >= 4
