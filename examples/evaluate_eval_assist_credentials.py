from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.eval_assist_constants import EvaluatorNameEnum, OptionSelectionStrategyEnum
from unitxt.eval_assist_utils import CatalogDefinition, parse_catalog_definition
from unitxt.loaders import LoadFromDictionary
from unitxt.text_utils import print_dict

logger = get_logger()

data = {
    "test": [
        {"context": {"Question": "How is the weather?"}},
    ]
}

criteria = "metrics.llm_as_judge.eval_assist.direct_assessment.criterias.temperature"

inference_engine_catalog_definition: CatalogDefinition = {
    'name': "engines.rits",
    'params': {
        'model_name': "mistralai/mistral-large-instruct-2407",
        "max_tokens": 1024,
        "seed": 42
    }
}

metric_catalog_definition: CatalogDefinition = {
    'name': 'metrics.llm_as_judge.eval_assist.direct_assessment',
    'params': {
        'inference_engine': inference_engine_catalog_definition,
        "option_selection_strategy": OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT.name,
        'evaluator_name': EvaluatorNameEnum.MIXTRAL_LARGE.name,
        'criteria_or_criterias': "metrics.llm_as_judge.eval_assist.direct_assessment.criterias.temperature"
    }
}

parsed_catalog_name = parse_catalog_definition(metric_catalog_definition)
print('parsed_catalog_name')
print(parsed_catalog_name)

card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    task=Task(
        input_fields={"context": dict},
        reference_fields={},
        prediction_type=str,
        metrics=[parsed_catalog_name],
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
]

evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
