from ..logging_utils import enable_explicit_format
from . import settings


def launch(demo_settings=None):
    enable_explicit_format()
    if demo_settings is not None:
        for key, value in demo_settings.items():
            setattr(settings, key, value)
    from .run import demo

    demo.launch(debug=True)


def launch_without_header(demo_settings=None):
    new_settings = {"HEADER_VISBLE": False}
    if demo_settings is not None:
        new_settings = {**new_settings, **demo_settings}
    launch(new_settings)
