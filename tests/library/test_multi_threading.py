from concurrent.futures import ThreadPoolExecutor

from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import TaskBasedLLMasJudge
from unitxt.operator import SequentialOperator
from unitxt.stream import MultiStream

from tests.utils import UnitxtTestCase

data = [
    {
        "question": "What is the capital of France?",
        "contexts": [
            "Paris is the capital of France.",
            "France is known for its culture and cuisine.",
        ],
        "answer": "Paris",
        "ground_truths": "Paris",
    },
    {
        "question": "Who wrote '1984'?",
        "contexts": [
            "George Orwell wrote the book 1984.",
            "1984 is a dystopian novel.",
        ],
        "answer": "George Orwell",
        "ground_truths": "George Orwell",
    },
    {
        "question": "What is the largest mammal?",
        "contexts": [
            "The blue whale is the largest mammal on Earth.",
            "Whales are large aquatic mammals.",
        ],
        "answer": "Elephant",
        "ground_truths": "Blue Whale",
    },
    {
        "question": "Where is the Great Wall located?",
        "contexts": [
            "The Great Wall of China is in Asia.",
            "It spans thousands of miles across northern China.",
        ],
        "answer": "China",
        "ground_truths": "India",
    },
    {
        "question": "What is the boiling point of water?",
        "contexts": [
            "Water boils at 100 degrees Celsius at sea level.",
            "The boiling point of water depends on pressure.",
        ],
        "answer": "100 degrees Celsius",
        "ground_truths": "100 degrees Celsius",
    },
    {
        "question": "Who developed the theory of relativity?",
        "contexts": [
            "Albert Einstein developed the theory of relativity.",
            "This theory changed modern physics.",
        ],
        "answer": "Albert Einstein",
        "ground_truths": "Albert Einstein",
    },
    {
        "question": "What element does 'O' represent on the periodic table?",
        "contexts": [
            "Oxygen is represented by 'O' on the periodic table.",
            "Oxygen is essential for respiration.",
        ],
        "answer": "Oxygen",
        "ground_truths": "Oxygen",
    },
    {
        "question": "What is the tallest mountain?",
        "contexts": [
            "Mount Everest is the tallest mountain above sea level.",
            "Located in the Himalayas.",
        ],
        "answer": "K2",
        "ground_truths": "Mount Everest",
    },
    {
        "question": "What gas do plants absorb?",
        "contexts": [
            "Plants absorb carbon dioxide during photosynthesis.",
            "They release oxygen as a byproduct.",
        ],
        "answer": "Carbon Dioxide",
        "ground_truths": "Carbon Dioxide",
    },
    {
        "question": "What is the main ingredient in bread?",
        "contexts": [
            "Flour is the main ingredient in most types of bread.",
            "Bread is made by baking a dough of flour and water.",
        ],
        "answer": "Water",
        "ground_truths": "Flour",
    },
]

METRIC_TO_TEMPLATE = {
    "faithfulness": "judge_with_question_simplified",
    "context_relevance": "judge_context_relevance_ares",
    "answer_correctness": "judge_loose_match_no_context",
    "answer_relevance": "judge_answer_relevance",
}


def get_llmaj_template(metric_name: str):
    template_name = METRIC_TO_TEMPLATE[metric_name]
    realization_suffix = "_verbal" if metric_name == "faithfulness" else "_numeric"
    return f"templates.rag_eval.{metric_name}.{template_name}{realization_suffix}"


class TestMultiThreading(UnitxtTestCase):
    def test_load_and_evaluate(self):
        model_id = "google/flan-ul2"
        metric_name = "context_relevance"

        inference_engine = CrossProviderInferenceEngine(
            model=model_id,
        )

        metric = TaskBasedLLMasJudge(
            inference_model=inference_engine,
            format=None,
            template=get_llmaj_template(metric_name=metric_name),
            task=f"tasks.rag_eval.{metric_name}.binary",
            main_score=f"{metric_name}_judge",
            prediction_field="contexts"
            if metric_name == "context_relevance"
            else "answer",
            infer_log_probs=False,
            judge_to_generator_fields_mapping={},
        )

        def evaluate_metrics(record):
            # record['contexts'] = transform_str_to_list(record["contexts"])
            multi_stream = MultiStream.from_iterables({"test": [record]}, copying=True)

            metrics_operator = SequentialOperator(steps=[metric])
            instances = list(metrics_operator(multi_stream)["test"])
            metric_values = []
            for i in range(len(instances)):
                metric_values.append(instances[i]["score"]["instance"]["score"])
            return metric_values

        with ThreadPoolExecutor(max_workers=2) as executer:
            results = executer.map(evaluate_metrics, data)
            results = list(results)

        self.assertListEqual(results, [[0.0]] * len(data))
