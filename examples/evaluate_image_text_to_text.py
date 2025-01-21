from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import LMMSEvalInferenceEngine
from unitxt.templates import MultipleChoiceTemplate, MultiReferenceTemplate


if True:
    import cvar_pyutils.debugging_tools
    cvar_pyutils.debugging_tools.set_remote_debugger('9.148.189.104', 55557) # 9.148.189.104
with settings.context(disable_hf_datasets_cache=False):
    inference_model = LMMSEvalInferenceEngine(
        model_type="llama_vision",
        model_args={"pretrained": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
        # model_type="llava",
        # model_args={"pretrained": "liuhaotian/llava-v1.5-7b"},
        max_new_tokens=512,
        image_token=""
    )
    # ai2d template
    # template = MultipleChoiceTemplate(
    #     input_format="{context}Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
    #     choices_separator="\n",
    #     target_field="answer",
    #     enumerator="capitals",
    # )
    # doc_vqa 11b/90b template
    template = MultiReferenceTemplate(
        input_format= "{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image." \
                      " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. " \
                      "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
        references_field="answers",
    )
    dataset = load_dataset(
        card="cards.doc_vqa.lmms_eval",
        format="formats.chat_api",
        split="test",
        template=template,
        max_test_instances=20,
        # metrics=["metrics.relaxed_correctness.json"]
    )
    # test_dataset = list(tqdm(dataset["test"], total=20))
    # test_dataset = list(tqdm(dataset["test"]))

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
    dataset = load_dataset(
        card="cards.doc_vqa.lmms_eval",
        template="templates.qa.with_context.lmms_eval",
        format="formats.models.llava_interleave",
        loader_limit=20,
        # augmentor="augmentors.image.grey_scale",
        augmentor="augmentors.image.rgb",
        streaming=True,
        metrics=["metrics.anls"],
        split="test",
        system_prompt="system_prompts.general.be_concise"
    )