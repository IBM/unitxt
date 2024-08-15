from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

logger = get_logger()
# First, we define the judge template.
judge_summary_rating_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's summary summarise well the given text.\n"
    'You must respond according the following format: "[[rate]] - explanation".\n'
    'Were the rate is a score between 0 to 10 (10 for great summary, 0 for a very poor one)".\n'
    "The explanation describe shortly why you decided to give the rank you chosen.\n"
    "Please make sure to start with your rank ([[rank]]) before anything else.\n"
    "For example: [[9]] The summary catches the main text ideas."
    ".\n\n",
    input_format="[Text:\n{question}\n\n" "Assistant's summary:\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

# Second, we define the inference engine we use for judge, with the preferred model and platform.
platform = "hf"
model_name = "google/flan-t5-large"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=256, use_fp16=True
)
# change to this to infer with IbmGenAI APIs:
#
# platform = 'ibm_gen_ai'
# model_name = 'meta-llama/llama-3-70b-instruct'
# inference_model = IbmGenAiInferenceEngine(model_name="meta-llama/llama-3-70b-instruct", max_new_tokens=512)

# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    inference_model=inference_model,
    template=judge_summary_rating_template,
    task="rating.single_turn",
    main_score=f"llm_judge_{model_name.split('/')[1].replace('-', '_')}_{platform}",
    strip_system_prompt_and_format_from_inputs=False,
)

# Load XSUM dataset, with the above metric.
dataset = load_dataset(
    card="cards.xsum",
    template="templates.summarization.abstractive.formal",
    metrics=[llm_judge_metric],
    loader_limit=5,
)

test_dataset = dataset["test"]

# Infer a model to get predictions.
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# Print results
print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "processed_prediction",
        "references",
        "score",
    ],
)


logger.info(
    "Now, we will repeat the example except this time we will use the reference for the judgement."
)

judge_summary_rating_with_reference_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's summary summarise well the given text.\n"
    "You will be given a reference answer and the assistant's answer."
    " Begin your evaluation by comparing the assistant's answer with the reference answer."
    " Identify and correct any mistakes."
    'You must respond according the following format: "[[rate]] - explanation".\n'
    'Were the rate is a score between 0 to 10 (10 for great summary, 0 for a very poor one)".\n'
    "The explanation describe shortly why you decided to give the rank you chosen.\n"
    "Please make sure to start with your rank ([[rank]]) before anything else.\n"
    "For example: [[9]] The summary catches the main text ideas."
    ".\n\n",
    input_format="[Text:\n{question}\n\n"
    "[The Start of Reference Summary]\n{reference_answer}\n[The End of Reference summary]\n\n"
    "[The Start of Assistant's summary]\n{answer}\n[The End of Assistant's summary]",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

llm_judge_with_summary_metric = LLMAsJudge(
    inference_model=inference_model,
    template=judge_summary_rating_with_reference_template,
    task="rating.single_turn_with_reference",
    main_score=f"llm_judge_{model_name.split('/')[1].replace('-', '_')}_{platform}",
    single_reference_per_prediction=True,
    strip_system_prompt_and_format_from_inputs=False,
)

# Load XSUM dataset, with the above metric.
dataset = load_dataset(
    card="cards.xsum",
    template="templates.summarization.abstractive.formal",
    metrics=[llm_judge_with_summary_metric],
    loader_limit=5,
)

test_dataset = dataset["test"]

# Infer a model to get predictions.
model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# Print results
print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "processed_prediction",
        "references",
        "score",
    ],
)
