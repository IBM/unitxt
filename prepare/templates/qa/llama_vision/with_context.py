from unitxt.artifact import fetch_artifact
from unitxt.catalog import add_to_catalog
from unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image."
                     " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. "
                     "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
        references_field="answers",
    ),
    "templates.qa.llama_vision.with_context.doc_vqa",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context} Read the text in the image carefully and answer the question with the text as seen exactly in the image."
                     " For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. "
                     "If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}",
        references_field="answers",
    ),
    "templates.qa.llama_vision.with_context.info_vqa",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context} {question}\nAnswer the question with a single word.",
        references_field="answers",
        __description__="lmms-evals default template for chartqa.",
    ),
    "templates.qa.llama_vision.with_context.chart_qa",
    overwrite=True,
)


