from typing import Any, Dict, List, Tuple, Union

from ..blocks import InputOutputTemplate
from .data_utils import SQLData


class Text2SQLInputOutputTemplate(InputOutputTemplate):
    """Input Output template for Text-to-SQL tasks.

    Attributes:
        instruction (str): Instruction prefix for the model. Defaults to "".
        num_samples (int): Number of samples to include from the tables in the prompt. Defaults to 0.
        use_schema_linking (bool): Whether to use schema linking (table and column mentions). Defaults to True.
        use_oracle_knowledge (bool): Whether to include oracle knowledge (evidence) in the prompt. Defaults to True.
        db_type (str): Type of the database. Defaults to "sqlite".
    """

    instruction: str = ""
    num_samples: int = 0
    # use_schema_linking: bool = True
    use_oracle_knowledge: bool = True
    db_type: str = "sqlite"

    def reference_fields_to_target_and_references(
        self, outputs: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        return (outputs["query"], [outputs["query"]])

    def input_fields_to_source(self, inputs: Dict[str, Any]) -> str:
        db_id: str = inputs["db_id"]
        question: str = inputs["utterance"]
        # if self.use_schema_linking:
        #     tables: Optional[List[str]] = inputs.get("table_mentions")
        #     columns: Optional[List[str]] = inputs.get("column_mentions")
        # else:
        #     tables, columns = None, None
        schema_text: str = SQLData().generate_schema_prompt(
            db_id,
            self.db_type,
            # tables,
            # columns,
            num_rows_from_table_to_add=self.num_samples,
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
            evidence: Union[str, List[str]] = inputs["evidence"]

        return self.apply_formatting(
            inputs,
            "input field",
            self.input_format,
            "input_format",
        )
