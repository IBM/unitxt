from unitxt import evaluate, load_dataset
from unitxt.inference import OpenAiInferenceEngineParams, InstructLabInferenceEngine, \
    NonBatchedInstructLabInferenceEngine
from unitxt.text_utils import print_dict

if __name__ == '__main__':
    dataset = load_dataset(
        card="cards.cnn_dailymail",
        template="templates.summarization.abstractive.instruct_full",
        loader_limit=100,
        # "metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn],
    )
    test_dataset = dataset['test']
    inference_model = NonBatchedInstructLabInferenceEngine(
        parameters=OpenAiInferenceEngineParams(max_tokens=1000),
        base_url = 'http://cccxc412.pok.ibm.com:9000/v1'
        )
    predictions = inference_model.infer(test_dataset)
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