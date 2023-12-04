import os
from contextlib import contextmanager


@contextmanager
def modified_environment(**additional_settings):
    # Save the original state
    original_state = os.environ.copy()

    # Merge new settings with the existing ones
    updated_state = {**original_state, **additional_settings}
    os.environ.update(updated_state)

    try:
        yield
    finally:
        # Revert to the original state
        os.environ.clear()
        os.environ.update(original_state)
