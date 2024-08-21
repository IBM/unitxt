from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Explode, Wrap
from unitxt.image_operators import ImageToText
from unitxt.operators import Copy
from unitxt.test_utils.card import test_card

for language in ["en", "fr"]:
    card = TaskCard(
        loader=LoadHF(path="cmarkea/doc-vqa"),
        preprocess_steps=[
            "splitters.small_no_dev",
            Explode(field=f"qa/{language}", to_field="pair"),
            Copy(field="pair/question", to_field="question"),
            Copy(field="pair/answer", to_field="answers"),
            Wrap(field="answers", inside="list"),
            ImageToText(field="image", to_field="context"),
            Set(fields={"context_type": "image"}),
        ],
        task="tasks.qa.with_context.abstractive",
        templates="templates.qa.with_context.all",
        __tags__={
            "license": "apache-2.0",
            "multilinguality": "monolingual",
            "modalities": ["image", "text"],
            "size_categories": "10K<n<100K",
            "task_categories": "question-answering",
            "task_ids": "extractive-qa",
        },
        __description__=(
            "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles. Supported Tasks and Leaderboards Question… See the full description on the dataset page: https://huggingface.co/datasets/rajpurkar/squad."
        ),
    )

    test_card(card)
    add_to_catalog(card, f"cards.doc_vqa.{language}", overwrite=True)
