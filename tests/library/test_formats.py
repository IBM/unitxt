from datetime import datetime

from unitxt.api import load_dataset
from unitxt.card import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.formats import (
    ChatAPIFormat,
    GraniteDocumentsFormat,
    HFSystemFormat,
    SystemFormat,
)
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Rename, Set
from unitxt.settings_utils import get_constants
from unitxt.standard import DatasetRecipe
from unitxt.system_prompts import TextualSystemPrompt
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate, MultiReferenceTemplate, TemplatesDict
from unitxt.test_utils.operators import (
    check_operator,
)

from tests.library.test_image_operators import create_random_jpeg_image
from tests.utils import UnitxtTestCase

# Assume
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
                "source": f"What is 3+2? <{constants.image_tag} src='media/images/0'>",
                "target": "5",
                "instruction": "solve the math exercises",
                "demos": demo_instances,
                "input_fields": {},
                "target_prefix": "The answer is ",
                "system_prompt": "You are a smart assistant.",
                "media": {
                    "images": [
                        {"image": create_random_jpeg_image(2, 2, 1), "format": "JPEG"}
                    ]
                },
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
                                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDO1G+u11O7AupwBM4AEh45NFFFaHxJ/9k=",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                "demos": demo_instances,
                "media": {"images": []},
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

    def test_granite_documents_format(self):
        inputs = [
            {
                "input_fields": {
                    "question": "what is love?",
                    "contexts": ["love is love"],
                },
            },
            {
                "input_fields": {
                    "question": "what is love?",
                    "context": "love is love",
                },
            },
        ]

        system_format = GraniteDocumentsFormat()

        today = datetime.today().strftime("%B %d, %Y")
        targets = [
            {
                "input_fields": {
                    "question": "what is love?",
                    "contexts": ["love is love"],
                },
                "source": "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
                + today
                + '.\nYou are Granite, developed by IBM. Write the response to the user\'s input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\n\nIn your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.<|end_of_text|>\n<|start_of_role|>documents<|end_of_role|>Document 0\nlove is love<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>what is love?<|end_of_text|>\n<|start_of_role|>assistant {"citations": true, "length": "long"}<|end_of_role|>',
            },
            {
                "input_fields": {
                    "question": "what is love?",
                    "context": "love is love",
                },
                "source": "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
                + today
                + '.\nYou are Granite, developed by IBM. Write the response to the user\'s input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\n\nIn your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.<|end_of_text|>\n<|start_of_role|>documents<|end_of_role|>Document 0\nlove is love<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>what is love?<|end_of_text|>\n<|start_of_role|>assistant {"citations": true, "length": "long"}<|end_of_role|>',
            },
        ]

        check_operator(
            operator=system_format,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        data = {
            "test": [
                {
                    "query": "What city is the largest in Texas?",
                    "extracted_chunks": "Austin is the capital of Texas.\nHouston is the the largest city in Texas but not the capital of it. ",
                    "expected_answer": "Houston",
                },
                {
                    "query": "What city is the capital of Texas?",
                    "extracted_chunks": "Houston is the the largest city in Texas but not the capital of it. ",
                    "expected_answer": "Austin",
                },
            ]
        }

        card = TaskCard(
            # Assumes this csv, contains 3 fields
            # question (string), extracted_chunks (string), expected_answer (string)
            loader=LoadFromDictionary(data=data),
            # Map these fields to the fields of the task.rag.response_generation task.
            # See https://www.unitxt.ai/en/latest/catalog/catalog.tasks.rag.response_generation.html
            preprocess_steps=[
                Rename(field_to_field={"query": "question"}),
                Wrap(field="extracted_chunks", inside="list", to_field="contexts"),
                Wrap(
                    field="expected_answer", inside="list", to_field="reference_answers"
                ),
                Set(
                    fields={
                        "contexts_ids": [],
                    }
                ),
            ],
            # Specify the task and the desired metrics (note that these are part of the default
            # metrics for the task, so the metrics selection can be omitted).
            task="tasks.rag.response_generation",
            # Specify a default template
            templates=TemplatesDict(
                {
                    "simple": MultiReferenceTemplate(
                        instruction="Answer the question based on the information provided in the document given below.\n\n",
                        input_format="Document: {contexts}\nQuestion: {question}",
                        references_field="reference_answers",
                    ),
                }
            ),
        )

        # select recommended metrics according to your available resources.
        metrics = [
            "metrics.rag.response_generation.recommended.cpu_only.all",
            # "metrics.rag.response_generation.recommended.small_llm.all",
            # "metrics.rag.response_generation.recommended.llmaj_watsonx.all",
            # "metrics.rag.response_generation.recommended.llmaj_rits.all"
            # "metrics.rag.response_generation.recommended.llmaj_azure.all"
        ]

        # Verbalize the dataset using the template
        dataset = load_dataset(
            card=card,
            template_card_index="simple",
            format=GraniteDocumentsFormat(),
            split="test",
            max_test_instances=10,
            metrics=metrics,
        )

        self.assertListEqual(
            list(dataset["source"]),
            [
                "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
                + today
                + '.\nYou are Granite, developed by IBM. Write the response to the user\'s input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\n\nIn your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.<|end_of_text|>\n<|start_of_role|>documents<|end_of_role|>Document 0\nAustin is the capital of Texas.\nHouston is the the largest city in Texas but not the capital of it. <|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What city is the largest in Texas?<|end_of_text|>\n<|start_of_role|>assistant {"citations": true, "length": "long"}<|end_of_role|>',
                "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
                + today
                + '.\nYou are Granite, developed by IBM. Write the response to the user\'s input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\n\nIn your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.<|end_of_text|>\n<|start_of_role|>documents<|end_of_role|>Document 0\nHouston is the the largest city in Texas but not the capital of it. <|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What city is the capital of Texas?<|end_of_text|>\n<|start_of_role|>assistant {"citations": true, "length": "long"}<|end_of_role|>',
            ],
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

        recipe = DatasetRecipe(
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
