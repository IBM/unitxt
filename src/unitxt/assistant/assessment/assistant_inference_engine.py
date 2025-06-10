import csv
import logging
import os
from typing import Any, Dict, List, Union

from unitxt import create_dataset, evaluate
from unitxt.assistant.app import Assistant
from unitxt.assistant.assessment.my_pretty_table import print_generic_table
from unitxt.dataset import Dataset
from unitxt.inference import InferenceEngine, TextGenerationInferenceOutput

logger = logging.getLogger("assistance-inference-engine")


class AssistantInferenceEngine(InferenceEngine):
    def prepare_engine(self):
        self.assistant = Assistant()

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], Dataset],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        sources = [x if isinstance(x, str) else x["source"] for x in dataset]
        messages = [[{"role": "user", "content": s}] for s in sources]
        generators = [self.assistant.generate_response(m) for m in messages]
        return ["".join(g) for g in generators]


if __name__ == "__main__":
    dataset_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "unitxt_assistant_qa_dataset.csv"
    )
    with open(dataset_file_path, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        dataset = list(reader)

    dataset = [
        {k: [v] if k == "answers" else v for k, v in line.items()} for line in dataset
    ]

    criteria = "metrics.llm_as_judge.direct.criteria.answer_completeness"
    metrics = [
        f"metrics.llm_as_judge.direct.rits.llama3_3_70b[criteria={criteria}, context_fields=[answers]]"
    ]

    dataset = create_dataset(
        task="tasks.qa.open",
        test_set=dataset,
        metrics=metrics,
    )
    dataset = dataset["test"]
    model = AssistantInferenceEngine()
    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    res_dict = [
        {
            "score": r["score"]["instance"]["score"],
            "source": r["source"],
            "prediction": r["prediction"],
            "target": r["target"],
            "judgement": results[0]["score"]["instance"][
                "answer_completeness_positional_bias_assessment"
            ],
        }
        for r in results
    ]
    col_width = {
        "score": 5,
        "source": 25,
        "prediction": 50,
        "target": 25,
        "judgement": 80,
    }
    print_generic_table(headers=col_width.keys(), data=res_dict, col_widths=col_width)

    logger.info("Global Scores:")
    logger.info(results.global_scores.summary)
