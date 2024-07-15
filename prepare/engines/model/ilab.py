from datasets import load_dataset

from unitxt import evaluate
from unitxt.inference import OpenAiInferenceEngineParams, InstructLabInferenceEngine
from unitxt.text_utils import print_dict

if __name__ == '__main__':
    test_dataset = load_dataset(
        "unitxt/data",
        "card=cards.squad,template=templates.qa.with_context.simple,"
        # "metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn],"
        "loader_limit=20",
        trust_remote_code=True,
        split="test",
    )
    inference_model = InstructLabInferenceEngine(
        # parameters=OpenAiInferenceEngineParams(),
        # base_url = 'http://127.0.0.1:8080/v1'
        )
    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
    for instance in evaluated_dataset:
        print(instance['score']['global'])
        break
        # print_dict(
        #     instance,
        #     keys_to_print=[
        #         "source",
        #         "prediction",
        #         "processed_prediction",
        #         "references",
        #         "score",
        #     ],
        # )