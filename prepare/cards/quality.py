import json

import requests

import unitxt
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import (
    MapInstanceValues,
    Set,
)
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card


def load_quality_split(split: str,
                       base_url: str = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1"):
    """Load and parse a specific split of the QuALITY dataset."""
    url = f"{base_url}/QuALITY.v1.0.1.htmlstripped.{split}"
    response = requests.get(url)
    response.raise_for_status()

    # Parse JSON Lines format and process the data
    processed_data = []
    for line in response.text.strip().split('\n'):
        if line:
            article_data = json.loads(line)

            # Extract common article information
            article_info = {
                'article_id': article_data['article_id'],
                'title': article_data['title'],
                'author': article_data.get('author', ''),
                'article': article_data['article'],
                'year': article_data.get('year', ''),
                'topic': article_data.get('topic', '')
            }

            # Process each question for this article
            for question in article_data.get('questions', []):
                # Use 'gold_label' as the answer and handle missing values
                gold_label = question.get('gold_label', None)
                if gold_label is None:
                    gold_label = -1
                    # continue

                processed_question = {
                    'context': article_info['article'],
                    'question': question['question'],
                    'question_id': question['question_unique_id'],
                    'choices': question['options'],
                    'answer': gold_label,  # Use the 'gold_label' as the answer
                    'difficulty': question.get('difficult', 0),
                    'title': article_info['title'],
                    'author': article_info['author'],
                    'year': article_info['year'],
                    'topic': article_info['topic']
                }
                processed_data.append(processed_question)

    return processed_data


def load_quality_data():
    """Load all splits of the QuALITY dataset."""
    data = {}
    for split_name, file_split in [('train', 'train'),  ('validation', 'dev')]:
        try:
            data[split_name] = load_quality_split(file_split)
        except Exception as e:
            print(f"Warning: Could not load {file_split} split: {e}")

    return data


# Create the card using LoadFromDictionary
with unitxt.settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadFromDictionary(
            data=load_quality_data(),
            data_classification_policy=["public"]
        ),
        preprocess_steps=[
            SplitRandomMix(
                {"train": "train[80%]", "validation": "validation[100%]", "test": "train[20%]"}
            ),
            MapInstanceValues(
                mappers={
                    "answer": {
                        "A": 0,
                        "B": 1,
                        "C": 2,
                        "D": 3,
                    }
                }
            ),
            Set(fields={"context_type": "document"}),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __description__=(
            """QuALITY (Question Answering with Long Input Texts, Yes!) is a multiple-choice reading comprehension dataset with long documents. The dataset comprises of documents from Project Gutenberg and questions written by human annotators. Each question has 4-5 answer choices, and requires understanding of the entire document to answer correctly. Questions are designed to test comprehensive understanding of the entire document, with various difficulty levels."""
        ),
        __tags__={
            "annotations_creators": "expert-generated",
            "language": ["en"],
            "license": "cc-by-4.0",
            "size_categories": ["10K<n<100K"],
            "task_categories": [
                "question-answering",
                "multiple-choice",
                "reading-comprehension"
            ],
            "multilinguality": "monolingual",
            "task_ids": [
                "extractive-qa",
                "reading-comprehension"
            ],
        },
    )

    # Test and add the card to the catalog
    test_card(card, strict=False)
    add_to_catalog(card, "cards.quality", overwrite=True)
