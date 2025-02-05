from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine, WMLInferenceEngineChat
from unitxt.templates import MultipleChoiceTemplate, MultiReferenceTemplate

import os
os.environ["WML_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WML_APIKEY"] = "64BP8yXXA-b4rcbFh0V_tAxuLUmsnr4DFA2h-WxU6xFk"
os.environ["WML_PROJECT_ID"] = "c7796a9e-7e37-478f-b2d7-140ed524c82c"
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_API_KEY"] = "64BP8yXXA-b4rcbFh0V_tAxuLUmsnr4DFA2h-WxU6xFk"
os.environ["WATSONX_PROJECT_ID"] = "c7796a9e-7e37-478f-b2d7-140ed524c82c"
# logging
if False:
    os.environ["LITELLM_LOG"] = "DEBUG"
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # additional settings for clean logs
    httpcore_logging = logging.getLogger("httpcore")
    httpcore_logging.setLevel(logging.ERROR)
    httpx_logging = logging.getLogger("httpx")
    httpx_logging.setLevel(logging.ERROR)
# debug
if False:
    import cvar_pyutils.debugging_tools
    cvar_pyutils.debugging_tools.set_remote_debugger('9.148.189.104', 55557) # 9.148.189.104
with settings.context(disable_hf_datasets_cache=False):


    # ai2d 90b template
    # template = MultipleChoiceTemplate(
    #     input_format="{context} Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
    #     choices_separator="\n",
    #     target_field="answer",
    #     enumerator="capitals",
    # )
    # max_tokens = 16
    # doc_vqa 11b/90b template
    template = MultiReferenceTemplate(
        input_format= "{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image." \
                      " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. " \
                      "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
        references_field="answers",
    )
    max_tokens = 32
    # chart_qa template
    # template = MultiReferenceTemplate(
    #     input_format="{context} {question}\nAnswer the question with a single word.",
    #     references_field="answers",
    #     __description__="lmms-evals default template for chartqa.",
    # )
    # max_tokens = 16
    dataset = load_dataset(
        card="cards.doc_vqa.lmms_eval",
        format="formats.chat_api",
        split="test",
        template=template,
        # max_test_instances=20,
        disable_cache=False
    )

    inference_model = CrossProviderInferenceEngine(model="llama-3-2-11b-vision-instruct", provider="watsonx",
                                                   max_tokens=max_tokens, temperature=0.0)
    # inference_model = WMLInferenceEngineChat(model_name="meta-llama/llama-3-2-11b-vision-instruct",
    #                                          max_tokens=max_tokens, temperature=0.0)

    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
