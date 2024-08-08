from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import ListFieldValues, Rename, Set
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()
orig_settings = settings.allow_unverified_code
settings.allow_unverified_code = True

for dataset_name in [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]:
    card = TaskCard(
        loader=LoadHF(
            path="heegyu/bbq", name=dataset_name, data_classification_policy=["public"]
        ),
        preprocess_steps=[
            Set({"context_type": "description"}),
            Rename(field_to_field={"label": "answer"}),
            ListFieldValues(fields=["ans0", "ans1", "ans2"], to_field="choices"),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __description__="Bias Benchmark for QA (BBQ), a dataset of question sets that highlight attested social biases against people belonging to protected classes along nine social dimensions relevant for U.S. English-speaking contexts.",
        __tags__={
            "languages": ["english"],
            "urls": {"arxiv": "https://arxiv.org/abs/2110.08193"},
        },
    )

    test_card(card, strict=False)
    add_to_catalog(card, "cards.safety.bbq." + dataset_name, overwrite=True)

settings.allow_unverified_code = orig_settings
