from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

logger = get_logger()

# First, we define the judge template.
judge_summary_rating_template = InputOutputTemplate(
    instruction=(
        "Please act as an impartial judge and evaluate if the assistant's summary summarise well the given text.\n"
        'You must respond according the following format: "[[rate]] - explanation".\n'
        'Were the rate is a score between 0 to 10 (10 for great summary, 0 for a very poor one)".\n'
        "The explanation describe shortly why you decided to give the rank you chosen.\n"
        "Please make sure to start with your rank ([[rank]]) before anything else.\n"
        "For example: [[9]] The summary catches the main text ideas."
        ".\n\n"
    ),
    input_format="[Text:\n{question}\n\n" "Assistant's summary:\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

# Second, we define the inference engine we use for judge, with the preferred model and provider.
# You can change the provider to any of: "watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"
model = CrossProviderInferenceEngine(model="llama-3-8b-instruct", provider="watsonx")

# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    model=model,
    template=judge_summary_rating_template,
    format="formats.chat_api",
    task="rating.single_turn",
    main_score="llm_judge_llama_3_8b",
    strip_system_prompt_and_format_from_inputs=False,
)

# Load XSUM dataset, with the above metric.
dataset = load_dataset(
    card="cards.xsum",
    template="templates.summarization.abstractive.formal",
    metrics=[llm_judge_metric],
    loader_limit=5,
    split="test",
)

# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = engine.infer(dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=dataset)

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
    model=model,
    template=judge_summary_rating_with_reference_template,
    task="rating.single_turn_with_reference",
    main_score="llm_judge_llama_3_2_1b_hf",
    single_reference_per_prediction=True,
    strip_system_prompt_and_format_from_inputs=False,
)

# Load XSUM dataset, with the above metric.
dataset = load_dataset(
    card="cards.xsum",
    template="templates.summarization.abstractive.formal",
    format="formats.chat_api",
    metrics=[llm_judge_with_summary_metric],
    loader_limit=5,
    split="test",
)

# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = engine.infer(dataset)

# Evaluate the predictions using the defined metric.
results = evaluate(predictions=predictions, data=dataset)

# Print results
print(
    results.instance_scores.to_df(
        columns=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
)
