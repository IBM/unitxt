from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    MapInstanceValues,
    RenameFields,
    SplitRandomMix,
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.test_utils.card import test_card

dataset_name = "CFPB"
subset_and_urls = {
    "watsonx": "https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/cfpb_complaints/cfpb_compliants.csv",
    "2023": "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_max=2023-01-04&date_received_min=2022-01-04&field=all&format=csv&has_narrative=true&lens=product&no_aggs=true&size=340390&sub_lens=sub_product&trend_depth=5&trend_interval=month",
}
field_to_field = {
    "watsonx": {"narrative": "text", "product": "label"},
    "2023": {"Consumer complaint narrative": "text", "Product": "label"},
}
mappers = {
    "watsonx": {
        "retail_banking": "retail banking",
        "mortgages_and_loans": "mortgages and loans",
        "debt_collection": "debt collection",
        "credit_card": "credit card",
        "credit_reporting": "credit reporting",
    },
    "2023": {
        "Credit reporting, credit repair services, or other personal consumer reports": "credit reporting or credit repair services or other personal consumer reports",
        "Credit card or prepaid card": "credit card or prepaid card",
        "Payday loan, title loan, or personal loan": "payday loan or title loan or personal loan",
        "Debt collection": "debt collection",
        "Mortgage": "mortgage",
        "Checking or savings account": "checking or savings account",
        "Money transfer, virtual currency, or money service": "money transfer or virtual currency or money service",
        "Vehicle loan or lease": "vehicle loan or lease",
        "Student loan": "student loan",
    },
}
for subset, url in subset_and_urls.items():
    card = TaskCard(
        loader=LoadCSV(files={"train": url}),
        preprocess_steps=[
            SplitRandomMix(
                {
                    "train": "train[70%]",
                    "validation": "train[10%]",
                    "test": "train[20%]",
                }
            ),
            RenameFields(field_to_field=field_to_field[subset]),
            MapInstanceValues(mappers={"label": mappers[subset]}),
            AddFields(
                fields={
                    "classes": list(mappers[subset].values()),
                    "text_type": "text",
                    "type_of_class": "topic",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
    )
    test_card(card, debug=False)
    add_to_catalog(card, f"cards.{dataset_name}.product.{subset}", overwrite=True)
