from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import DatasetRecipe
from unitxt.templates import MultipleChoiceTemplate, MultiReferenceTemplate

ai2d_llama_vision_template = MultipleChoiceTemplate(
    input_format="{context} Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
    choices_separator="\n",
    target_field="answer",
    enumerator="capitals",
)
doc_vqa_llama_vision_template = MultiReferenceTemplate(
    input_format="{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image."
                 " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. "
                 "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
    references_field="answers",
)
chart_qa_llama_vision_template = MultiReferenceTemplate(
    input_format="{context} {question}\nAnswer the question with a single word.",
    references_field="answers",
    __description__="lmms-evals default template for chartqa.",
)
benchmark = Benchmark(
    subsets={
        "doc_vqa_default": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
        ),
        "info_vqa_default": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
        ),
        "chart_qa_default": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
        ),
        "ai2d_default": DatasetRecipe(
            card="cards.ai2d",
        ),
        "websrc_default": DatasetRecipe(
            card="cards.websrc",
        ),
        "doc_vqa_llama_vision_template": DatasetRecipe(
            card="cards.doc_vqa.lmms_eval",
            template=doc_vqa_llama_vision_template,
            format="formats.chat_api",
        ),
        "info_vqa_llama_vision_template": DatasetRecipe(
            card="cards.info_vqa_lmms_eval",
            template=doc_vqa_llama_vision_template,
            format="formats.chat_api",
        ),
        "chart_qa_llama_vision_template": DatasetRecipe(
            card="cards.chart_qa_lmms_eval",
            template=chart_qa_llama_vision_template,
            format="formats.chat_api",
        ),
        "ai2d_llama_vision_template": DatasetRecipe(
            card="cards.ai2d",
            template=ai2d_llama_vision_template,
            format="formats.chat_api",
        ),
    },
)

add_to_catalog(benchmark, "benchmarks.vision_full", overwrite=True)
