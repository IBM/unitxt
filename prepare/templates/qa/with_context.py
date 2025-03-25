from unitxt.catalog import add_to_catalog
from unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Context: {context}\nQuestion: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.simple",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="based on this text: {context}\n answer the question: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.simple2",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="Based on this {context_type}:\n {context}\n answer the question: {question}",
        references_field="answers",
    ),
    "templates.qa.with_context.with_type",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{question}\nAnswer based on this {context_type}:\n {context}",
        references_field="answers",
    ),
    "templates.qa.with_context.question_first",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question based on the information provided in the document given below. The answer should be a single word or a number or a short phrase of few words.",
        input_format="Document: {context}\nQuestion:{question}",
        output_format="{answer}",
        target_prefix="Answer:\n",
        references_field="answers",
        __description__="Template from https://huggingface.co/datasets/abacusai/WikiQA-Free_Form_QA",
    ),
    "templates.qa.with_context.ffqa",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        __deprecated_msg__="This template should be replaced with `templates.qa.with_context` as it adds an unnecessary instruction to the model to return a short answer.",
        instruction="Answer the question based on the information provided in the {context_type} given below. The answer should be a single word or a number or a short phrase of few words.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}",
        output_format="{answer}",
        target_prefix="Answer:\n",
        references_field="answers",
        title_fields=["context_type"],
    ),
    "templates.qa.with_context.title",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question based on the information provided in the given {context_type}.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}",
        output_format="{answer}",
        target_prefix="Answer:\n",
        references_field="answers",
        title_fields=["context_type"],
    ),
    "templates.qa.with_context",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Answer the question directly based on the information provided in the {context_type}. Extract the exact phrase from the {context_type} that directly answers the question, without any alterations.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}",
        output_format="{answer}",
        target_prefix="Answer:\n",
        references_field="answers",
        title_fields=["context_type"],
    ),
    "templates.qa.extractive",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Using the information from the {context_type} given below, summarize a paragraph-long response to the following user query."
        + "\nHere are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.",
        input_format="{context_type}:\n{context}\nQuery:\n{question}",
        output_format="{answers}",
        target_prefix="Answer:\n",
        references_field="answers",
        title_fields=["context_type"],
        __description__="Template from https://arxiv.org/pdf/2305.14303 for query-focused summarization over tables",
    ),
    "templates.qa.with_context.qtsumm",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word or phrase.",
        references_field="answers",
    ),
    "templates.qa.with_context.lmms_eval",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.with_context",
            "templates.qa.extractive",
            "templates.qa.with_context.simple",
            "templates.qa.with_context.simple2",
            "templates.qa.with_context.with_type",
            "templates.qa.with_context.question_first",
            "templates.qa.with_context.ffqa",
            "templates.qa.with_context.title",
            "templates.qa.with_context.lmms_eval",
        ]
    ),
    "templates.qa.with_context.all",
    overwrite=True,
)

add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word.",
        references_field="answers",
        __description__="lmms-evals default template for chartqa.",
    ),
    "templates.qa.with_context.chart_qa",
    overwrite=True,
)


add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word or phrase.",
        references_field="answers",
        __description__="lmms-evals default template for docvqa.",
    ),
    "templates.qa.with_context.doc_vqa",
    overwrite=True,
)
add_to_catalog(
    MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word or phrase.",
        references_field="answers",
        __description__="lmms-evals default template for docvqa.",
    ),
    "templates.qa.with_context.info_vqa",
    overwrite=True,
)