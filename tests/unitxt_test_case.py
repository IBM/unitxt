from src.unitxt.test_utils.catalog import add_local_catalog_to_artifactories_env_var


def setUpClass(cls):
    add_local_catalog_to_artifactories_env_var()


def setup_unitxt_test_env(cls):
    setattr(cls, setUpClass.__name__, classmethod(setUpClass))
    return cls
