import os

current_dir = os.path.dirname(os.path.abspath(__file__))
app_file = os.path.join(current_dir, "app.py")


def launch():
    from streamlit.web import cli

    cli.main_run([app_file])
