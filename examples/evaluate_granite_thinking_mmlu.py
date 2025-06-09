from unitxt.api import evaluate, load_dataset
from unitxt.formats import HFSystemFormat
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.templates import MultipleChoiceTemplate

for thinking in [True, False]:
    postprocessors = ["processors.first_character"]
    if thinking:
        postprocessors = [
            PostProcess(
                ExtractWithRegex(regex="<response>(.*)</response"),
                process_references=False,
            ),
            "processors.first_character",
        ]

    template = MultipleChoiceTemplate(
        input_format="""The following are multiple choice questions (with answers) about {topic}.
    {question}
    Answers:
    {choices}
    The response should be returned as a single letter: A, B, C, or D. Do not answer in sentences. Only return the single letter answer.""",
        target_field="answer",
        choices_separator="\n",
        postprocessors=postprocessors,
    )
    dataset = load_dataset(
        card="cards.mmlu.abstract_algebra",
        template=template,
        split="test",
        format=HFSystemFormat(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            chat_kwargs_dict={"thinking": thinking},
            place_instruction_in_user_turns=True,
        ),
        loader_limit=100,
    )

    model = CrossProviderInferenceEngine(
        model="granite-3-3-8b-instruct", provider="rits", temperature=0
    )

    predictions = model(dataset)

    results = evaluate(predictions=predictions, data=dataset)

    print("Instance Results when Thinking=", thinking)

    for instance in results.instance_scores:
        if instance["processed_prediction"] not in ["A", "B", "C", "D"]:
            print(
                "Problematic prediction (could not be parsed to a acceptable single letter answer):"
            )
            print(instance["prediction"])

    print("Global Results when Thinking=", thinking)
    print(results.global_scores.summary)
