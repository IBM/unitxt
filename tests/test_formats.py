import unittest

from src.unitxt.formats import ICLFormat
from src.unitxt.operators import ModelInputFormatter


class TestFormats(unittest.TestCase):
    def test_icl_format_with_demonstrations(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
        )

        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        result = format.format(instance, demos_instances=demos_instances)
        target = """Instruction:solve the math exercises

User:1+2
Agent: 3

User:4-2
Agent: 2

User:1+1
Agent:"""
        self.assertEqual(result, target)

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent: {target}\n\n",
            model_input_format="Instruction:{instruction}{demos}User:{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction, and add demos into it:
        instance["instruction"] = "solve the math exercises"
        instance["demos"] = demos_instances

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_with_demonstrations_and_instruction_after_demos(self):
        iclformat = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
            add_instruction_at_start=False,
            add_instruction_after_demos=True,
        )

        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        result = iclformat.format(instance, demos_instances=demos_instances)
        target = """User:1+2
Agent: 3

User:4-2
Agent: 2

User:solve the math exercises

1+1
Agent:"""
        self.assertEqual(result, target)

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent: {target}\n\n",
            model_input_format="{demos}User:{instruction}{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction, and add demos into it:
        instance["instruction"] = "solve the math exercises"
        instance["demos"] = demos_instances

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
        )

        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
        }

        result = format.format(instance)
        target = """Instruction:solve the math exercises

User:1+1
Agent:"""
        self.assertEqual(result, target)

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent: {target}\n\n",
            model_input_format="Instruction:{instruction}{demos}User:{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction
        instance["instruction"] = "solve the math exercises"

        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations_or_instruction(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
        )

        instance = {"source": "1+1", "target": "2"}

        result = format.format(instance)
        target = """User:1+1
Agent:"""
        self.assertEqual(result, target)

        # compare here with ModelInputFormatter
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent: {target}\n\n",
            model_input_format="{instruction}{demos}User:{source}\nAgent:",
        )
        # no need to refresh instance, no instructions and no demo.
        instance_out = model_input_formatter.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_icl_format_without_demonstrations_and_empty_instruction(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
        )

        instance = {"source": "1+1", "target": "2", "instruction": ""}

        result = format.format(instance)
        target = """User:1+1
Agent:"""
        self.assertEqual(result, target)
