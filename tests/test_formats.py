import unittest

from src.unitxt.operators import ModelInputFormatter


class TestFormats(unittest.TestCase):
    def test_icl_format_with_demonstrations(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        target = """Instruction:solve the math exercises

User:1+2
Agent:3

User:4-2
Agent:2

User:1+1
Agent:"""

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction, and add demos into it:
        instance["instruction"] = "solve the math exercises"
        instance["demos"] = demos_instances

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_with_demonstrations_and_instruction_after_demos(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        target = """User:1+2
Agent:3

User:4-2
Agent:2

User:solve the math exercises

1+1
Agent:"""
        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{demos}User:{instruction}\n\n{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction, and add demos into it:
        instance["instruction"] = "solve the math exercises"
        instance["demos"] = demos_instances

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }

        target = """Instruction:solve the math exercises

User:1+1
Agent:"""

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction
        instance["instruction"] = "solve the math exercises"

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations_or_instruction(self):
        instance = {"source": "1+1", "target": "2"}
        target = """User:1+1
Agent:"""

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{instruction}{demos}User:{source}\nAgent:",
        )
        # no need to refresh instance, no instructions and no demo.
        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations_and_empty_instruction(self):
        instance = {"source": "1+1", "target": "2", "instruction": ""}

        target = """User:1+1
Agent:"""

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{instruction}{demos}User:{source}\nAgent:",
        )
        # no need to refresh instance, no instructions and no demo.
        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)
