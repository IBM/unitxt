import unittest

from src.unitxt.formats import ICLFormat


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
            "instruction": "solve the math excercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        result = format.format(instance, demos_instances=demos_instances)
        target = """Instruction:solve the math excercises

User:1+2
Agent: 3

User:4-2
Agent: 2

User:1+1
Agent:"""
        self.assertEqual(result, target)

    def test_icl_format_with_demonstrations_and_instruction_after_demos(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
            add_instruction_at_start=False,
            add_instruction_after_demos=True,
        )

        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math excercises",
        }
        demos_instances = [
            {"source": "1+2", "target": "3"},
            {"source": "4-2", "target": "2"},
        ]

        format.format(instance, demos_instances=demos_instances)

    def test_icl_format_without_demonstrations(self):
        format = ICLFormat(
            input_prefix="User:",
            output_prefix="Agent:",
            instruction_prefix="Instruction:",
        )

        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math excercises",
        }

        result = format.format(instance)
        target = """Instruction:solve the math excercises

User:1+1
Agent:"""
        self.assertEqual(result, target)

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
