from unitxt.formats import HFSystemFormat, SystemFormat
from unitxt.test_utils.operators import (
    check_operator,
)

from tests.utils import UnitxtTestCase


class TestFormats(UnitxtTestCase):
    def test_hf_system_format(self):
        instruction = "solve the math exercises"

        demo_instances = [
            {"source": "1+2", "target": "3", "instruction": instruction, "inputs": {}},
            {"source": "4-2", "target": "2", "instruction": instruction, "inputs": {}},
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
        ]

        # imitating iclformat's add_instruction_after_demos=True, instruction is not "", and target_prefix =""
        system_format = HFSystemFormat(model_name="HuggingFaceH4/zephyr-7b-beta")

        targets = [
            {
                "target": "2",
                "inputs": {},
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s> \n<|user|>\n1+2</s> \n<|assistant|>\nThe answer is 3</s> \n<|user|>\n4-2</s> \n<|assistant|>\nThe answer is 2</s> \n<|user|>\n1+1</s> \n<|assistant|>\nThe answer is ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s> \n<|user|>\n1+2</s> \n<|assistant|>\nThe answer is 3</s> \n<|user|>\n4-2</s> \n<|assistant|>\nThe answer is 2</s> \n<|user|>\n3+2</s> \n<|assistant|>\nThe answer is ",
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_system_format(self):
        instruction = "solve the math exercises"

        demo_instances = [
            {"source": "1+2", "target": "3", "instruction": instruction, "inputs": {}},
            {"source": "4-2", "target": "2", "instruction": instruction, "inputs": {}},
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "7-4",
                "target": "3",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "12-3",
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
                "target": "2",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n1+1\nAgent: ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n3+2\nAgent: ",
            },
            {
                "target": "3",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n7-4\nAgent: ",
            },
            {
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
                "target": "2",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "target": "3",
                "inputs": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
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
        for demo_instance in inputs[0]["demos"]:
            demo_instance.pop("instruction")

        system_format = SystemFormat(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="{demos}User: {instruction}{source}\nAgent: ",
        )

        targets_no_instruction = [
            {
                "target": "2",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "target": "3",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
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
                    "instruction": "classify user sentence by its sentiment to either positive, or negative.",
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "instruction": "classify user sentence by its sentiment to either positive, or negative.",
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
                    "instruction": "classify user sentence by its sentiment to either positive, or negative.",
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "instruction": "classify user sentence by its sentiment to either positive, or negative.",
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

    def test_system_format_with_args(self):
        system_format = SystemFormat(
            format_args={"input_prefix": "User: ", "output_prefix": "Agent: "},
            demos_field="demos",
            demo_format="{input_prefix}{source}\n{output_prefix}{target}\n\n",
            model_input_format="{demos}{input_prefix}{instruction}\n\n{source}\n{output_prefix}",
        )

        instruction = "solve the math exercises"

        demo_instances = [
            {"source": "1+2", "target": "3", "instruction": instruction, "inputs": {}},
            {"source": "4-2", "target": "2", "instruction": instruction, "inputs": {}},
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "7-4",
                "target": "3",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
            {
                "source": "12-3",
                "target": "9",
                "instruction": instruction,
                "demos": demo_instances,
                "inputs": {},
            },
        ]

        targets = [
            {
                "target": "2",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n1+1\nAgent: ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n3+2\nAgent: ",
            },
            {
                "target": "3",
                "inputs": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n7-4\nAgent: ",
            },
            {
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
