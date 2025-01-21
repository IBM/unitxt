from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
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
    inference_model = CrossProviderInferenceEngine(model="llama-3-2-11b-vision-instruct", provider="watsonx")

    # # ai2d 11b template chain of thought
    # template = MultipleChoiceTemplate(
    #     input_format="{context}Look at the scientific diagram carefully and answer the following question: "
    #                  "{question}\n{choices}\n Think step by step and finally respond to the question with only the"
    #                  " correct option number as 'FINAL ANSWER'. Last token needs to the answer number Let's think step by step.",
    #     choices_separator="\n",
    #     target_field="answer",
    #     enumerator="capitals",
    # )
    # # ai2d 90b template
    # template = MultipleChoiceTemplate(
    #     input_format="{context}Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
    #     choices_separator="\n",
    #     target_field="answer",
    #     enumerator="capitals",
    # )
    # doc_vqa 11b/90b template
    # template = MultiReferenceTemplate(
    #     input_format= "{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image." \
    #                   " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. " \
    #                   "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
    #     references_field="answers",
    # )
    # websrc template
    template = MultiReferenceTemplate(
        input_format="{context}\nAnswer the question using a single word or phrase.\n{question}",
        references_field="answers",
        __description__="lmms-evals default template for websrc.",
    )
    # # chart_qa template
    # template = MultiReferenceTemplate(
    #     input_format="{context}\n{question}\nAnswer the question using a single word.",
    #     references_field="answers",
    #     __description__="lmms-evals default template for chartqa.",
    # )
    dataset = load_dataset(
        card="cards.websrc",
        format="formats.chat_api",
        split="test",
        template=template,
        # max_test_instances=20,
    )


    predictions = inference_model.infer(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
