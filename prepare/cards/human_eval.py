from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import ExecuteExpression
from unitxt.settings_utils import get_settings
from unitxt.test_utils.card import test_card

settings = get_settings()
orig_settings = settings.allow_unverified_code
settings.allow_unverified_code = True


get_asserts = '[t for t in re.findall(r"assert.*?(?=\\n\\s*assert|$)", test.replace("candidate", entry_point), re.DOTALL)]'


card = TaskCard(
    loader=LoadHF(path="openai_humaneval", split="test"),
    preprocess_steps=[
        ExecuteExpression(
            expression=get_asserts, imports_list=["re"], to_field="test_list"
        )
    ],
    task=Task(
        inputs=["prompt"],
        outputs=["prompt", "canonical_solution", "test_list"],
        metrics=["metrics.bleu"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{prompt}\n",
                output_format="{prompt}\n{canonical_solution}",
            ),
        ]
    ),
    __tags__={
        "annotations_creators": "expert-generated",
        "arxiv": "2107.03374",
        "flags": ["code-generation"],
        "language": "en",
        "language_creators": "expert-generated",
        "license": "mit",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "n<1K",
        "source_datasets": "original",
        "task_categories": "text2text-generation",
    },
    __description__=(
        "Dataset Card for OpenAI HumanEval Dataset Summary The HumanEval dataset released by OpenAI includes 164 programming problems with a function sig- nature, docstring, body, and several unit tests. They were handwritten to ensure not to be included in the training set of code generation models. Supported Tasks and Leaderboards Languages The programming problems are written in Python and contain English natural text in comments and… See the full description on the dataset page: https://huggingface.co/datasets/openai_humaneval."
    ),
)

test_card(
    card,
    demos_taken_from="test",
    demos_pool_size=1,
    num_demos=0,
    strict=False,
    loader_limit=500,
    debug=False,
)
add_to_catalog(card, "cards.human_eval", overwrite=True)

settings.allow_unverified_code = orig_settings
