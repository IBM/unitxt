import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from unitxt.artifact import (
    Artifact,
    MissingArtifactTypeError,
)
from unitxt.logging_utils import get_logger

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestArtifactRecovery(UnitxtTestCase):
    def test_custom_catalog_and_project(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            project_dir = Path(tmpdirname)
            operator_dir = project_dir / "operators"
            catalog_dir = project_dir / "catalog"
            operator_dir.mkdir()

            # Write the operator class
            operator_code = textwrap.dedent(
                """
                from unitxt.operators import InstanceOperator

                class MyTempOperator(InstanceOperator):
                    def process(self, instance, stream_name=None):
                        return instance
            """
            )
            (operator_dir / "my_operator.py").write_text(operator_code)
            (operator_dir / "__init__.py").write_text("")

            # Write the saving script
            saving_code = textwrap.dedent(
                f"""
                from operators.my_operator import MyTempOperator
                from unitxt import add_to_catalog, settings

                add_to_catalog(MyTempOperator(), "operators.my_temp_operator", catalog_path="{catalog_dir}")
            """
            )
            saving_script = project_dir / "save_operator.py"
            saving_script.write_text(saving_code)

            # Write the loading script
            loading_code = textwrap.dedent(
                """
                from unitxt import get_from_catalog
                from operators.my_operator import MyTempOperator

                get_from_catalog("operators.my_temp_operator")
            """
            )
            loading_script = project_dir / "load_operator.py"
            loading_script.write_text(loading_code)

            # Run the saving script
            result_save = subprocess.run(
                [sys.executable, str(saving_script)],
                env={
                    "UNITXT_CATALOGS": str(catalog_dir),
                    "PYTHONPATH": str(project_dir),
                },
                capture_output=True,
                text=True,
            )
            if result_save.returncode != 0:
                logger.info(f"Saving script STDOUT:\n{result_save.stdout}")
                logger.info(f"Saving script STDERR:\n{result_save.stderr}")
            self.assertEqual(result_save.returncode, 0, "Saving script failed")

            # Run the loading script
            result_load = subprocess.run(
                [sys.executable, str(loading_script)],
                env={
                    "UNITXT_CATALOGS": str(catalog_dir),
                    "PYTHONPATH": str(project_dir),
                },
                capture_output=True,
                text=True,
            )
            if result_load.returncode != 0:
                logger.info(f"Loading script STDOUT:\n{result_load.stdout}")
                logger.info(f"Loading script STDERR:\n{result_load.stderr}")
            self.assertEqual(result_load.returncode, 0, "Loading script failed")

    def test_correct_artifact_recovery(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = Artifact.from_dict(args)
        self.assertEqual(a.num_demos, 0)

    def test_correct_artifact_recovery_with_overwrite(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 0,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        a = Artifact.from_dict(args, overwrite_args={"num_demos": 1})
        self.assertEqual(a.num_demos, 1)

    def test_bad_artifact_recovery_missing_type(self):
        args = {
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(MissingArtifactTypeError):
            Artifact.from_dict(args)

    def test_bad_artifact_recovery_bad_type(self):
        args = {
            "__type__": "dataset_recipe",
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        with self.assertRaises(ValueError):
            Artifact.from_dict(args)

        try:
            Artifact.from_dict(args)
        except Exception as e:
            logger.info(e)

    def test_subclass_registration_and_loading(self):
        args = {
            "__type__": "dummy_not_exist",
        }
        with self.assertRaises(ValueError):
            Artifact.from_dict(args)

        try:
            Artifact.from_dict(args)
        except ValueError as e:
            logger.info("The error message (not a real error):", e)

        class DummyExistForLoading(Artifact):
            pass

        args = {
            "__type__": "dummy_exist_for_loading",
        }
        Artifact.from_dict(args)
