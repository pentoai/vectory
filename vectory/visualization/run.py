from pathlib import Path

from streamlit import config
from streamlit.bootstrap import run


def run_streamlit():
    app_path = Path(__file__).parent / "main.py"

    config.set_option("server.headless", True)
    run(str(app_path), "", [], [])
