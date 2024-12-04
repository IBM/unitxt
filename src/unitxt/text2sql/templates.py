from typing import Dict, List, Tuple

from ..blocks import InputOutputTemplate
from .data_utils import SQLData


class Text2SQLInputOutputTemplate(InputOutputTemplate):
    instruction: str = ""
    num_samples: int = 0
    use_schema_linking: bool = True
    use_oracle_knowledge: bool = True
    db_type: str = "sqlite"

    def reference_fields_to_target_and_references(
        self, outputs: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        return (outputs["sql"], [outputs["sql"]])

    def input_fields_to_source(self, inputs: Dict[str, object]) -> str:
        db_id = inputs["db_id"]
        question = inputs["question"]
        if self.use_schema_linking:
            tables = inputs.get("table_mentions", None)
            columns = inputs.get("column_mentions", None)
        schema_text = SQLData().db_to_schema_text(
            db_id, self.db_type, tables, columns, num_rows=self.num_samples
        )

        inputs.update(
            {
                "instruction": self.instruction,
                "question": question,
                "db_id": db_id,
                "schema_text": schema_text,
            }
        )
        if self.use_oracle_knowledge:
            evidence = inputs["evidence"]
            evidence = "; ".join(evidence) if isinstance(evidence, list) else evidence
            inputs["evidence"] = evidence

        return self.apply_formatting(
            inputs,
            "input field",
            self.input_format,
            "input_format",
        )
