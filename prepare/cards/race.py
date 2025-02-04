from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import IndexOf, Rename, Set
from unitxt.test_utils.card import test_card

numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for subset in ["all", "high", "middle"]:
    card = TaskCard(
        loader=LoadHF(path="race", name=subset),
        preprocess_steps=[
            Set({"numbering": numbering}),
            IndexOf(search_in="numbering", index_of="answer", to_field="answer"),
            Rename(
                field_to_field={"options": "choices", "article": "context"},
            ),
            Set({"context_type": "article"}),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __tags__={
            "annotations_creators": "expert-generated",
            "arxiv": "1704.04683",
            "language": "en",
            "language_creators": "found",
            "license": "other",
            "multilinguality": "monolingual",
            "region": "us",
            "size_categories": "10K<n<100K",
            "source_datasets": "original",
            "task_categories": "multiple-choice",
            "task_ids": "multiple-choice-qa",
        },
        __description__=(
            "RACE is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students. The dataset can be served as the training and test sets for machine comprehension. Supported Tasks and Leaderboards More Information Needed Languages More… See the full description on the dataset page: https://huggingface.co/datasets/ehovy/race."
        ),
    )
    if subset == "middle":
        test_card(card, strict=False)
    add_to_catalog(card, f"cards.race_{subset}", overwrite=True)
