import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from unitxt import settings
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    VLLMInferenceEngine,
)
from unitxt.templates import MultipleChoiceTemplate, MultiReferenceTemplate

if False:
    import cvar_pyutils.debugging_tools
    cvar_pyutils.debugging_tools.set_remote_debugger('9.148.189.104', 55557) # 9.148.189.104
with settings.context(
    disable_hf_datasets_cache=False,
):

    # ai2d 90b template
    # template = MultipleChoiceTemplate(
    #     input_format="{context} Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
    #     choices_separator="\n",
    #     target_field="answer",
    #     enumerator="capitals",
    # )
    # max_tokens = 16
    # doc_vqa 11b/90b template
    # template = MultiReferenceTemplate(
    #     input_format= "{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image." \
    #                   " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. " \
    #                   "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
    #     references_field="answers",
    # )
    # max_tokens = 32
    # chart_qa template
    template = MultiReferenceTemplate(
        input_format="{context} {question}\nAnswer the question with a single word.",
        references_field="answers",
        __description__="lmms-evals default template for chartqa.",
    )
    max_tokens = 16
    dataset = load_dataset(
        card="cards.chart_qa_lmms_eval",
        format="formats.chat_api",
        template=template,
        # loader_limit=2,
        split="test",
        disable_cache=False
    )

    inference_model = VLLMInferenceEngine(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        max_tokens=max_tokens, temperature=0.0
    )

    predictions = inference_model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
