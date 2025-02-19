from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.operators import Copy, ExecuteExpression
from unitxt.templates import NullTemplate

for subset in [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]:
    card = TaskCard(
        loader=LoadHF(
            path="rungalileo/ragbench",
            name=subset,
            split="test"
        ),
        preprocess_steps=[
            Copy(field="response", to_field="answer"),
            Copy(field="documents", to_field="contexts"),
            ExecuteExpression(expression="int(adherence_score)", to_field="number_val"),
            ExecuteExpression(expression="['yes' if adherence_score else 'no']", to_field="is_faithful"),
        ],
        task="tasks.rag_eval.faithfulness.binary",
        templates={"default": NullTemplate()},
    )

    add_to_catalog(
        card, f"cards.rag_eval.faithfulness.ragbench.{subset}", overwrite=True
    )
