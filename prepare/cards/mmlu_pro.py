import datasets
from unitxt.blocks import LoadHF, RenameFields, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import FilterByExpression
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card


def main():
    ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    topics = list(set(ds["test"]["category"]))

    for topic in topics:
        card = TaskCard(
            loader=LoadHF(path="TIGER-Lab/MMLU-Pro"),
            preprocess_steps=[
                FilterByExpression(f"category == '{topic}'"),
                RenameSplits({"validation": "train"}),
                RenameFields(
                    field_to_field={
                        "options": "choices",
                        "answer_index": "answer",
                    }
                ),
                Set(
                    fields={
                        "topic": topic,
                    }
                ),
            ],
            task="tasks.qa.multiple_choice.with_topic",
            templates="templates.qa.multiple_choice.with_topic.all",
            # __tags__={
            #     "annotations_creators": "no-annotation",
            #     "arxiv": ["2009.03300", "2005.00700", "2005.14165", "2008.02275"],
            #     "language": "en",
            #     "language_creators": "expert-generated",
            #     "license": "mit",
            #     "multilinguality": "monolingual",
            #     "region": "us",
            #     "size_categories": "10K<n<100K",
            #     "source_datasets": "original",
            #     "task_categories": "question-answering",
            #     "task_ids": "multiple-choice-qa",
            # },
            # __description__=(
            #     "Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021). \n"
            #     "This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57… See the full description on the dataset page: https://huggingface.co/datasets/cais/mmlu."
            # ),
        )
        if topic == topics[0]:
            test_card(
                card,
                strict=False,  # random generation here does not produce 0 results (MCQA)
                loader_limit=2 * len(ds["test"]),
            )
        add_to_catalog(card, f"cards.mmlu_pro.{topic.replace(' ','_')}", overwrite=True)


main()
