from ..logging_utils import enable_explicit_format
from ..settings_utils import get_settings
from . import settings


def launch(demo_settings=None):
    enable_explicit_format()
    if demo_settings is not None:
        for key, value in demo_settings.items():
            setattr(settings, key, value)
    from .run import demo

    unitxt_settings = get_settings()
    unitxt_settings.global_loader_limit = 300
    unitxt_settings.disable_hf_datasets_cache = False

    demo.launch(debug=True)


def launch_without_header(demo_settings=None):
    new_settings = {"HEADER_VISIBLE": False}
    if demo_settings is not None:
        new_settings = {**new_settings, **demo_settings}
    launch(new_settings)
