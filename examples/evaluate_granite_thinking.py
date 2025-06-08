from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.formats import HFSystemFormat
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# Set up question answer pairs in a dictionary
test_set = [
    {
        "question": "If I had 32 apples, I lost 5 apples, and gain twice more as many as I have.  How many do I have at the end",
        "answer": "81",
    },
]


# define the QA task
task = Task(
    input_fields={"question": str},
    reference_fields={"answer": str},
    prediction_type=str,
    metrics=["metrics.accuracy"],
)


# Create a simple template that formats the input.
# Add lowercase normalization as a post processor.


for thinking in [True, False]:
    postprocessors = ["processors.lower_case"]
    if thinking:
        postprocessors.append(
            PostProcess(
                ExtractWithRegex(regex="<response>(.*)</response"),
                process_references=False,
            )
        )

    template = InputOutputTemplate(
        instruction="Answer the following question with the single numeric answer.",
        input_format="{question}",
        output_format="{answer}",
        postprocessors=postprocessors,
    )
    dataset = create_dataset(
        task=task,
        test_set=test_set,
        template=template,
        split="test",
        format=HFSystemFormat(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            chat_kwargs_dict={"thinking": thinking},
            place_instruction_in_user_turns=True,
        ),
    )

    model = CrossProviderInferenceEngine(
        model="granite-3-3-8b-instruct", provider="rits", use_cache=False
    )

    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    print("Instance Results when Thinking=", thinking)
    print(results.instance_scores)
