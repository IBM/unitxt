import sys

from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Rename, Set, Shuffle
from unitxt.string_operators import FormatText
from unitxt.test_utils.card import test_card

task_cfgs = {
    # "A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word. Label the type of mark for the following products."
    "abercrombie": {
        "non_task_entries": {
            "label_field_name": "answer",
            "text_field_name": "text",
        },
        "classes_descriptions": "A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word.",
        "type_of_class": "type of mark",
        "text_type": "products",
        "classes": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "instruction": "{classes_descriptions}\n\nLabel the {type_of_class} for the following {text_type}:\n",
        "input_format": "Q: {text} What is the {type_of_class}?",
        "target_prefix": "A: ",
    },
    # A private right of action is when a regular person, a private citizen, is legally entitled to enforce their rights under a given statute. Does the clause specify a private right of action? Answer Yes or No.
    # https://github.com/HazyResearch/legalbench/blob/main/tasks/proa/base_prompt.txt
    "proa": {
        "non_task_entries": {
            "label_field_name": "answer",
            "text_field_name": "text",
        },
        "classes_descriptions": "A private right of action is when a regular person, a private citizen, is legally entitled to enforce their rights under a given statute",
        "type_of_class": "a private right of action",
        "text_type": "clause",
        "instruction": "{classes_descriptions}. Does the {text_type} specify {type_of_class}? Answer from one of {classes}",
        "classes": ["Yes", "No"],
        "title_fields": ["text_type"],
        "input_format": "{text_type}: {text}",
        "target_prefix": "A: ",
    },
    #    Classify the following text using the following definitions.\n\n- Facts: The paragraph describes the factual background that led up to the present lawsuit.\n- Procedural History: The paragraph describes the course of litigation that led to the current proceeding before the court.\n- Issue: The paragraph describes the legal or factual issue that must be resolved by the court.\n- Rule: The paragraph describes a rule of law relevant to resolving the issue.\n- Analysis: The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.\n- Conclusion: The paragraph presents a conclusion of the court.\n- Decree: The paragraph constitutes a decree resolving the dispute."
    # https://github.com/HazyResearch/legalbench/blob/main/tasks/function_of_decision_section/base_prompt.txt
    "function_of_decision_section": {
        "non_task_entries": {
            "label_field_name": "answer",
            "text_field_name": "Paragraph",
        },
        "classes_descriptions": "- Facts: The paragraph describes the factual background that led up to the present lawsuit.\n- Procedural History: The paragraph describes the course of litigation that led to the current proceeding before the court.\n- Issue: The paragraph describes the legal or factual issue that must be resolved by the court.\n- Rule: The paragraph describes a rule of law relevant to resolving the issue.\n- Analysis: The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.\n- Conclusion: The paragraph presents a conclusion of the court.\n- Decree: The paragraph constitutes a decree resolving the dispute",
        "type_of_class": "",
        "text_type": "text",
        "classes": [
            "Facts",
            "Procedural History",
            "Issue",
            "Rule",
            "Analysis",
            "Conclusion",
            "Decree",
        ],
        "instruction": "Classify the following {text_type} using the following definitions.\n\n{classes_descriptions}.\n\n",
        "title_fields": ["text_type"],
        "input_format": "{text_type}: {text}",
        "target_prefix": "Label: ",
    },
    # "Answer the following questions considering the state of international law on January 1st, 2020. Answer Yes or No."
    # https://github.com/HazyResearch/legalbench/blob/main/tasks/international_citizenship_questions/base_prompt.txt
    "international_citizenship_questions": {
        "non_task_entries": {
            "label_field_name": "answer",
            "text_field_name": "question",
        },
        "classes_descriptions": "considering the state of international law on January 1st, 2020",
        "type_of_class": "",
        "text_type": "question",
        "title_fields": ["text_type"],
        "instruction": "Answer the following {text_type} {classes_descriptions}.\n",
        "classes": ["Yes", "No"],
        "input_format": "{text_type}: {text} Answer from one of {classes}.",
        "target_prefix": "Answer: ",
    },
    # You are a lobbyist analyzing Congressional bills for their impacts on companies. Given the title and summary of the bill, plus information on the company from its 10K SEC filing, it is your job to determine if a bill is at least somewhat relevant to a company in terms of whether it could impact the company's bottom-line if it was enacted (by saying Yes or No).
    # https://github.com/HazyResearch/legalbench/blob/main/tasks/corporate_lobbying/base_prompt.txt
    "corporate_lobbying": {
        "non_task_entries": {
            "label_field_name": "answer",
            "text_field_name": "text",
            "text_verbalizer": "Official title of bill: {bill_title}\nOfficial summary of bill: {bill_summary}\nCompany name: {company_name}\nCompany business description: {company_description}",
        },
        "classes_descriptions": "You are a lobbyist analyzing Congressional bills for their impacts on companies.\nGiven the title and summary of the bill, plus information on the company from its 10K SEC filing, is a bill at least somewhat relevant to a company in terms of whether it could impact the company's bottom-line if it was enacted",
        "type_of_class": "",
        "text_type": "",
        "instruction": "{classes_descriptions}, it is your job to determine {type_of_class} (by saying Yes or No).",
        "classes": ["Yes", "No"],
        "input_format": "{text}\nIs this bill potentially relevant to the company? FINAL ANSWER:",
    },
}

for task_name, task_cfg in task_cfgs.items():
    card = TaskCard(
        loader=LoadHF(path="nguha/legalbench", name=task_name),
        preprocess_steps=(
            [Shuffle(page_size=sys.maxsize)]
            + (
                [
                    FormatText(
                        text=task_cfg["non_task_entries"]["text_verbalizer"],
                        to_field="text",
                    )
                ]
                if task_cfg["non_task_entries"].get("text_verbalizer", False)
                else []
            )
            + [
                Rename(
                    field_to_field={
                        task_cfg["non_task_entries"]["text_field_name"]: "text",
                        task_cfg["non_task_entries"]["label_field_name"]: "label",
                    }
                ),
                Set(
                    fields={
                        "text_type": task_cfg["text_type"],
                        "classes": task_cfg["classes"],
                        "type_of_class": task_cfg["type_of_class"],
                        "classes_descriptions": task_cfg["classes_descriptions"],
                    }
                ),
            ]
        ),
        task="tasks.classification.multi_class.with_classes_descriptions",
        templates={
            "default": InputOutputTemplate(
                input_format=task_cfg["input_format"],
                output_format="{label}",
                instruction=task_cfg["instruction"],
                target_prefix=task_cfg.get("target_prefix", ""),
                title_fields=task_cfg.get("title_fields", []),
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            ),
        },
        __tags__={
            "arxiv": "2308.11462",
            "flags": ["finance", "law", "legal"],
            "language": "en",
            "license": "other",
            "region": "us",
            "size_categories": "10K<n<100K",
            "task_categories": [
                "text-classification",
                "question-answering",
                "text-generation",
            ],
        },
        __description__=(
            "LegalBench is a collection of benchmark tasks for evaluating legal reasoning in large language models… See the full description on the dataset page: https://huggingface.co/datasets/nguha/legalbench"
        ),
    )

    test_card(card, format="formats.textual_assistant")
    add_to_catalog(card, f"cards.legalbench.{task_name}", overwrite=True)


# from unitxt import load_dataset
# ds = load_dataset("card=cards.legalbench.proa,template_card_index=default")
