from unitxt.artifact import fetch_artifact
from unitxt.card import TaskCard
from unitxt.formats import HFSystemFormat, SystemFormat
from unitxt.loaders import LoadHF
from unitxt.operators import ExecuteExpression, FilterByExpression, Set
from unitxt.settings_utils import get_settings
from unitxt.splitters import RenameSplits
from unitxt.standard import StandardRecipe
from unitxt.templates import MultipleChoiceTemplate
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
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s>\n<|user|>\n1+2</s>\n<|assistant|>\nThe answer is 3</s>\n<|user|>\n4-2</s>\n<|assistant|>\nThe answer is 2</s>\n<|user|>\n1+1</s>\n<|assistant|>\nThe answer is ",
            },
            {
                "target": "5",
                "inputs": {},
                "source": "<|system|>\nYou are a smart assistant.\nsolve the math exercises</s>\n<|user|>\n1+2</s>\n<|assistant|>\nThe answer is 3</s>\n<|user|>\n4-2</s>\n<|assistant|>\nThe answer is 2</s>\n<|user|>\n3+2</s>\n<|assistant|>\nThe answer is ",
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

    # flake8: noqa: RUF001
    def test_system_format_with_demos_different_target_prefixes(self):
        settings = get_settings()
        settings.allow_unverified_code = True

        task, _ = fetch_artifact("tasks.qa.multiple_choice.with_topic")
        task.input_fields["core_of_question"] = "str"

        template = MultipleChoiceTemplate(
            input_format="The following are multiple choice questions (with answers) about {topic}.\n{question}\nPotential Answers:\n{choices}\nAnswer:",
            target_field="answer",
            postprocessors=["processors.first_character"],
            choices_separator="\n",
            target_prefix="The {core_of_question} is: ",
        )

        card = TaskCard(
            loader=LoadHF(path="cais/mmlu", name="elementary_mathematics"),
            preprocess_steps=[
                RenameSplits({"test": "train"}),
                Set({"topic": "elementary mathematics"}),
                FilterByExpression("'what is the' in question.lower()"),
                ExecuteExpression(
                    to_field="core_of_question",
                    expression='question[12+question.lower().index("what is the "):].replace("?","")',
                ),
            ],
            # task="tasks.qa.multiple_choice.with_topic",
            task=task,
            # templates=["templates.qa.multiple_choice.with_topic.mmlu"],
            templates=[template],
        )

        recipe = StandardRecipe(
            card=card,
            loader_limit=500,
            demos_pool_size=5,
            num_demos=3,
            template_card_index=0,
        )
        ms = recipe()
        trains = list(ms["train"])

        formatted_source = trains[0]["source"]
        target_formatted_source = (
            "The following are multiple choice questions (with answers) about elementary mathematics.\n"
            "What is the remainder of 21 divided by 7?\n"
            "Potential Answers:\n"
            "A. 21\n"
            "B. 7\n"
            "C. 1\n"
            "D. None of these\n"
            "Answer:\n"
            "The remainder of 21 divided by 7 is: D\n"
            "\n"
            "The following are multiple choice questions (with answers) about elementary mathematics.\n"
            "Use the expression below to answer the question. 3 × [(2 × 6 – 5) + (8 ÷ 4)] – 1 What is the value of the expression?\n"
            "Potential Answers:\n"
            "A. 9\n"
            "B. 11\n"
            "C. 26\n"
            "D. 32\n"
            "Answer:\n"
            "The value of the expression is: C\n"
            "\n"
            "The following are multiple choice questions (with answers) about elementary mathematics.\n"
            "A soccer team has $90.00 to buy soccer balls. If one soccer ball costs $15.60, what is the greatest number of soccer balls the team can buy?\n"
            "Potential Answers:\n"
            "A. 4\n"
            "B. 5\n"
            "C. 6\n"
            "D. 7\n"
            "Answer:\n"
            "The greatest number of soccer balls the team can buy is: B\n"
            "\n"
            "The following are multiple choice questions (with answers) about elementary mathematics.\n"
            "What is the quotient for the expression 2,314 / 4?\n"
            "Potential Answers:\n"
            "A. 508\n"
            "B. 508 r2\n"
            "C. 578\n"
            "D. 578 r2\n"
            "Answer:\n"
            "The quotient for the expression 2,314 / 4 is: "
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
