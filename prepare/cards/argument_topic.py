from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.test_utils.card import test_card

dataset_name = "argument_topic"

class_names = [
    "affirmative action",
    "algorithmic trading",
    "assisted suicide",
    "atheism",
    "austerity regime",
    "blockade of the gaza strip",
    "cancel pride parades",
    "cannabis",
    "capital punishment",
    "collectivism",
    "compulsory voting",
    "cosmetic surgery",
    "cosmetic surgery for minors",
    "embryonic stem cell research",
    "entrapment",
    "executive compensation",
    "factory farming",
    "fast food",
    "fight urbanization",
    "flag burning",
    "foster care",
    "gender-neutral language",
    "guantanamo bay detention camp",
    "holocaust denial",
    "homeopathy",
    "homeschooling",
    "human cloning",
    "intellectual property rights",
    "intelligence tests",
    "journalism",
    "judicial activism",
    "libertarianism",
    "marriage",
    "missionary work",
    "multi-party system",
    "naturopathy",
    "organ trade",
    "payday loans",
    "polygamy",
    "private military companies",
    "prostitution",
    "racial profiling",
    "retirement",
    "safe spaces",
    "school prayer",
    "sex selection",
    "social media",
    "space exploration",
    "stay-at-home dads",
    "student loans",
    "surrogacy",
    "targeted killing",
    "telemarketing",
    "television",
    "the abolition of nuclear weapons",
    "the church of scientology",
    "the development of autonomous cars",
    "the olympic games",
    "the right to keep and bear arms",
    "the three-strikes laws",
    "the use of child actors",
    "the use of economic sanctions",
    "the use of public defenders",
    "the use of school uniform",
    "the vow of celibacy",
    "vocational education",
    "whaling",
    "wikipedia",
    "women in combat",
    "zero-tolerance policy in schools",
    "zoos",
]

card = TaskCard(
    loader=LoadHF(path="ibm/argument_quality_ranking_30k", name=f"{dataset_name}"),
    preprocess_steps=[
        Set(
            fields={
                "classes": class_names,
                "text_type": "argument",  # TODO maybe text?
            }
        ),
    ],
    task="tasks.classification.multi_class.topic_classification",
    templates="templates.classification.multi_class.all",
    __tags__={
        "arxiv": "1911.11408",
        "language": "en",
        "license": "cc-by-3.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-classification",
    },
    __description__=(
        "Argument Quality Ranking\n"
        "The dataset contains 30,497 crowd-sourced arguments for 71 debatable topics labeled for quality and stance, split into train, validation and test sets. The dataset was originally published as part of our paper: A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis.\n"
        "Argument Topic\n"
        "This subset contains 9,487 of the… See the full description on the dataset page: https://huggingface.co/datasets/ibm/argument_quality_ranking_30k."
    ),
)

test_card(card, debug=False)
add_to_catalog(artifact=card, name=f"cards.{dataset_name}", overwrite=True)
