from unitxt.card import TaskCard
from unitxt.formats import ChatAPIFormat, HFSystemFormat, SystemFormat
from unitxt.loaders import LoadFromDictionary
from unitxt.settings_utils import get_constants
from unitxt.standard import StandardRecipe
from unitxt.system_prompts import TextualSystemPrompt
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.operators import (
    check_operator,
)

from tests.utils import UnitxtTestCase

constants = get_constants()


class TestFormats(UnitxtTestCase):
    def test_openai_format(self):
        instruction = "solve the math exercises"

        demo_instances = [
            {
                "source": "1+2",
                "target": "3",
                "instruction": instruction,
                "input_fields": {},
            },
            {
                "source": "4-2",
                "target": "2",
                "instruction": instruction,
                "input_fields": {},
            },
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
        ]

        # imitating iclformat's add_instruction_after_demos=True, instruction is not "", and target_prefix =""
        system_format = ChatAPIFormat()

        targets = [
            {
                "target": "2",
                "input_fields": {},
                "source": [
                    {
                        "role": "system",
                        "content": "You are a smart assistant.\nsolve the math exercises",
                    },
                    {"role": "user", "content": "1+2"},
                    {"role": "assistant", "content": "The answer is 3"},
                    {"role": "user", "content": "4-2"},
                    {"role": "assistant", "content": "The answer is 2"},
                    {"role": "user", "content": "1+1"},
                ],
                "demos": demo_instances,
            },
            {
                "target": "5",
                "input_fields": {},
                "source": [
                    {
                        "role": "system",
                        "content": "You are a smart assistant.\nsolve the math exercises",
                    },
                    {"role": "user", "content": "1+2"},
                    {"role": "assistant", "content": "The answer is 3"},
                    {"role": "user", "content": "4-2"},
                    {"role": "assistant", "content": "The answer is 2"},
                    {"role": "user", "content": "3+2"},
                ],
                "demos": demo_instances,
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_openai_format_with_images(self):
        demo_instances = [
            {
                "source": f"What is 1+2? <{constants.image_tag} src='https://example.com/image1.jpg'>",
                "target": "3",
                "instruction": "solve the math exercises",
                "input_fields": {},
            },
            {
                "source": f"What is 4-2? <{constants.image_tag} src='https://example.com/image2.jpg'>",
                "target": "2",
                "instruction": "solve the math exercises",
                "input_fields": {},
            },
        ]

        inputs = [
            {
                "source": f"What is 1+1? <{constants.image_tag} src='https://example.com/image3.jpg'>",
                "target": "2",
                "instruction": "solve the math exercises",
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
            {
                "source": f"What is 3+2? <{constants.image_tag} src='https://example.com/image4.jpg'>",
                "target": "5",
                "instruction": "solve the math exercises",
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
        ]

        # Expected structured format for targets after processing
        targets = [
            {
                "target": "2",
                "input_fields": {},
                "source": [
                    {
                        "role": "system",
                        "content": "You are a smart assistant.\nsolve the math exercises",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 1+2? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image1.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                    {"role": "assistant", "content": "The answer is 3"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 4-2? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image2.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                    {"role": "assistant", "content": "The answer is 2"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 1+1? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image3.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                "demos": demo_instances,
            },
            {
                "target": "5",
                "input_fields": {},
                "source": [
                    {
                        "role": "system",
                        "content": "You are a smart assistant.\nsolve the math exercises",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 1+2? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image1.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                    {"role": "assistant", "content": "The answer is 3"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 4-2? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image2.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                    {"role": "assistant", "content": "The answer is 2"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 3+2? "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image4.jpg",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                "demos": demo_instances,
            },
        ]

        # Imitating iclformat's add_instruction_after_demos=True, instruction is not "", and target_prefix = ""
        system_format = ChatAPIFormat()

        # Run the check
        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_hf_system_format(self):
        instruction = "solve the math exercises"

        demo_instances = [
            {
                "source": "1+2",
                "target": "3",
                "instruction": instruction,
                "input_fields": {},
            },
            {
                "source": "4-2",
                "target": "2",
                "instruction": instruction,
                "input_fields": {},
            },
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
            },
        ]

        # imitating iclformat's add_instruction_after_demos=True, instruction is not "", and target_prefix =""
        system_format = HFSystemFormat(model_name="HuggingFaceH4/zephyr-7b-beta")

        targets = [
            {
                "target": "2",
                "input_fields": {},
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s>\n<|user|>\n1+2</s>\n<|assistant|>\nThe answer is 3</s>\n<|user|>\n4-2</s>\n<|assistant|>\nThe answer is 2</s>\n<|user|>\n1+1</s>\n<|assistant|>\nThe answer is ",
                "demos": demo_instances,
            },
            {
                "target": "5",
                "input_fields": {},
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s>\n<|user|>\n1+2</s>\n<|assistant|>\nThe answer is 3</s>\n<|user|>\n4-2</s>\n<|assistant|>\nThe answer is 2</s>\n<|user|>\n3+2</s>\n<|assistant|>\nThe answer is ",
                "demos": demo_instances,
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
            {
                "source": "1+2",
                "target": "3",
                "instruction": instruction,
                "input_fields": {},
            },
            {
                "source": "4-2",
                "target": "2",
                "instruction": instruction,
                "input_fields": {},
            },
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "7-4",
                "target": "3",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "12-3",
                "target": "9",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
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
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n1+1\nAgent: ",
                "demos": demo_instances,
            },
            {
                "target": "5",
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n3+2\nAgent: ",
                "demos": demo_instances,
            },
            {
                "target": "3",
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n7-4\nAgent: ",
                "demos": demo_instances,
            },
            {
                "target": "9",
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n12-3\nAgent: ",
                "demos": demo_instances,
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
                "demos": demo_instances,
                "input_fields": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "target": "5",
                "demos": demo_instances,
                "input_fields": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "target": "3",
                "demos": demo_instances,
                "input_fields": {},
                "source": "Instruction: solve the math exercises\n\nUser: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
                "target": "9",
                "demos": demo_instances,
                "input_fields": {},
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
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 1+1\nAgent: ",
            },
            {
                "target": "5",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 3+2\nAgent: ",
            },
            {
                "target": "3",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: 7-4\nAgent: ",
            },
            {
                "target": "9",
                "demos": demo_instances,
                "input_fields": {},
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
            "input_fields": {},
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
            "input_fields": {},
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

        self.assertDictEqual(result, target)

        # no demos
        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "input_fields": {},
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
            "input_fields": {},
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
            "input_fields": {},
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
            "input_fields": {},
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

        self.assertDictEqual(result, target)

    def test_system_format_with_demos_different_target_prefixes(self):
        instances = [
            {"question": "1+1", "answer": "2"},
            {"question": "2+2", "answer": "4"},
            {"question": "3+3", "answer": "6"},
            {"question": "4+4", "answer": "8"},
            {"question": "5+5", "answer": "10"},
            {"question": "6+6", "answer": "12"},
            {"question": "7+7", "answer": "14"},
            {"question": "8+8", "answer": "16"},
            {"question": "9+9", "answer": "18"},
            {"question": "10+10", "answer": "20"},
        ]

        task = Task(
            input_fields={"question": str},
            reference_fields={"answer": str},
            prediction_type=str,
            metrics=["metrics.accuracy"],
        )

        template = InputOutputTemplate(
            input_format="Solve: {question}\nAnswer: ",
            output_format="{answer}",
            postprocessors=[],
            target_prefix="{question} = ",
        )

        card = TaskCard(
            loader=LoadFromDictionary(data={"train": instances}),
            preprocess_steps=[],
            task=task,
            templates=[template],
        )

        recipe = StandardRecipe(
            card=card,
            loader_limit=20,
            demos_pool_size=5,
            num_demos=2,
            template_card_index=0,
            system_prompt=TextualSystemPrompt("\nSolve the following exercises.\n "),
        )
        ms = recipe()
        trains = list(ms["train"])

        formatted_source = (
            trains[0]["source"]
            + "\n\n"
            + trains[1]["source"]
            + "\n\n"
            + trains[2]["source"]
        )
        target_formatted_source = (
            "\n"
            "Solve the following exercises.\n"
            " \n"
            "Solve: 4+4\n"
            "Answer: \n"
            "4+4 = 8\n"
            "\n"
            "Solve: 3+3\n"
            "Answer: \n"
            "3+3 = 6\n"
            "\n"
            "Solve: 6+6\n"
            "Answer: \n"
            "6+6 = \n"
            "\n"
            "\n"
            "Solve the following exercises.\n"
            " \n"
            "Solve: 3+3\n"
            "Answer: \n"
            "3+3 = 6\n"
            "\n"
            "Solve: 4+4\n"
            "Answer: \n"
            "4+4 = 8\n"
            "\n"
            "Solve: 7+7\n"
            "Answer: \n"
            "7+7 = \n"
            "\n"
            "\n"
            "Solve the following exercises.\n"
            " \n"
            "Solve: 4+4\n"
            "Answer: \n"
            "4+4 = 8\n"
            "\n"
            "Solve: 5+5\n"
            "Answer: \n"
            "5+5 = 10\n"
            "\n"
            "Solve: 8+8\n"
            "Answer: \n"
            "8+8 = "
        )
        self.assertEqual(target_formatted_source, formatted_source)

    def test_system_format_with_args(self):
        system_format = SystemFormat(
            format_args={"input_prefix": "User: ", "output_prefix": "Agent: "},
            demos_field="demos",
            demo_format="{input_prefix}{source}\n{output_prefix}{target}\n\n",
            model_input_format="{demos}{input_prefix}{instruction}\n\n{source}\n{output_prefix}",
        )

        instruction = "solve the math exercises"

        demo_instances = [
            {
                "source": "1+2",
                "target": "3",
                "instruction": instruction,
                "input_fields": {},
            },
            {
                "source": "4-2",
                "target": "2",
                "instruction": instruction,
                "input_fields": {},
            },
        ]

        inputs = [
            {
                "source": "1+1",
                "target": "2",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "3+2",
                "target": "5",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "7-4",
                "target": "3",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
            {
                "source": "12-3",
                "target": "9",
                "instruction": instruction,
                "demos": demo_instances,
                "input_fields": {},
            },
        ]

        targets = [
            {
                "target": "2",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n1+1\nAgent: ",
            },
            {
                "target": "5",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n3+2\nAgent: ",
            },
            {
                "target": "3",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n7-4\nAgent: ",
            },
            {
                "target": "9",
                "demos": demo_instances,
                "input_fields": {},
                "source": "User: 1+2\nAgent: 3\n\nUser: 4-2\nAgent: 2\n\nUser: solve the math exercises\n\n12-3\nAgent: ",
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )
