from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Copy
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="squad"),
    preprocess_steps=[
        "splitters.small_no_test",
        Copy(field="answers/text", to_field="answers"),
        Set({"context_type": "passage"}),
    ],
    task="tasks.qa.with_context[metrics=[metrics.squad]]",
    templates="templates.qa.with_context.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1606.05250",
        "language": "en",
        "language_creators": ["crowdsourced", "found"],
        "license": "cc-by-sa-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended|wikipedia",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles. Supported Tasks and Leaderboards Question… See the full description on the dataset page: https://huggingface.co/datasets/rajpurkar/squad."
    ),
)

test_card(card)
add_to_catalog(card, "cards.squad", overwrite=True)
