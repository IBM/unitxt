from src.unitxt.formats import SystemFormat
from src.unitxt.test_utils.operators import (
    check_operator,
)
from tests.utils import UnitxtTestCase


class TestOperators(UnitxtTestCase):
    def test_system_format(self):
        demo_instances = [
            {"source": "1+2", "target": "3", "inputs": {}},
            {"source": "4-2", "target": "2", "inputs": {}},
        ]
        instruction = "solve the math exercises"

        inputs = [
            {
                "source": "1+1",
                "source1": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "3+2",
                "source1": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "7-4",
                "source1": "7-4",
                "target": "3",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "12-3",
                "source1": "12-3",
                "target": "9",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
        ]

        # imitating iclformat's add_instruction_after_demos=True, instruction is not "", and target_prefix =""
        system_format = SystemFormat(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="{demos}User: {instruction}\n\n{source}\nAgent: ",
        )

        targets = [
            {
                "source1": "1+1",
                "target": "2",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n1+1\nAgent: ",
            },
            {
                "source1": "3+2",
                "target": "5",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n3+2\nAgent: ",
            },
            {
                "source1": "7-4",
                "target": "3",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n7-4\nAgent: ",
            },
            {
                "source1": "12-3",
                "target": "9",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n12-3\nAgent: ",
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        # now imitate instruction before demos.
        system_format = SystemFormat(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="Instruction: {instruction}\n\n{demos}User: {source}\nAgent: ",
        )

        targets = [
            {
                "source1": "1+1",
                "target": "2",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "source1": "3+2",
                "target": "5",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "source1": "7-4",
                "target": "3",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
                "source1": "12-3",
                "target": "9",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 12-3\nAgent: ",
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        # test with instruction = "":
        for instance in inputs:
            instance.pop("instruction")

        system_format = SystemFormat(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="{demos}User: {instruction}{source}\nAgent: ",
        )

        targets_no_instruction = [
            {
                "source1": "1+1",
                "target": "2",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "source1": "3+2",
                "target": "5",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "source1": "7-4",
                "target": "3",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
                "source1": "12-3",
                "target": "9",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 12-3\nAgent: ",
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets_no_instruction,
            tester=self,
        )

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "inputs": {},
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "target": "positive",
                    "references": ["positive"],
                },
            ],
        }

        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )

        result = system_format.process(instance)

        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so not good"\nAgent:negative\n\nUser:This is my sentence: "was so good"\nAgent:positive\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
            "inputs": {},
        }
        self.assertDictEqual(result, target)

        # no demos
        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "inputs": {},
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
        }
        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )
        result = system_format.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "inputs": {},
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

        # test_system_format_with_prefix_and_suffix(self):
        system_format_fix = SystemFormat(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="[INST] <<SYS>>\n{instruction}\n\n{demos}User: {source}\nAgent: [/INST]",
        )
        renderer = system_format_fix

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "inputs": {},
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "target": "positive",
                    "references": ["positive"],
                },
            ],
        }
        self.maxDiff = None
        result = renderer.process(instance)
        target = {
            "source": '[INST] <<SYS>>\nclassify user sentence by its sentiment to either positive, or negative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: [/INST]',
            "target": "negative",
            "references": ["negative"],
            "inputs": {},
        }

        self.assertDictEqual(result, target)
