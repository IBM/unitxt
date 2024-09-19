import pandas as pd
from unitxt.artifact import fetch_artifact
from unitxt.catalog import add_to_catalog
from unitxt.templates import MultipleChoiceTemplate, TemplatesList

templates = {
    "with_topic": {
        "mmlu": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n{question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n{question}.\n選択肢: {choices}.\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n\\{question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n{question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n{question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n\n{question}.\nAatworten: {choices}.\nAatwort:",
        },
        "helm": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n\n質問: {question}.\n選択肢: {choices}.\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n\nPergunta: {question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n\nPregunta: {question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n\nQuestion: {question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n\nFrage: {question}.\nAatworten: {choices}.\nAatwort:",
        },
        "lm_eval_harness": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n\n{question}\n{choices}\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n\n{question}.\n{choices}\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n\n{question}.\n{choices}\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n\n{question}.\n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n\n{question}.\n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n\n{question}.\n{choices}.\nAatwort:",
        },
    },
    "open": {
        "helm": {
            "en": "Question: {question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "質問: {question}.\n選択肢: \n{choices}.\n答え:",
            "pt": "Pergunta: {question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Pregunta: {question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Question: {question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Frage: {question}.\nAatworten: {choices}.\nAatwort:",
        },
        "lm_eval_harness": {
            "en": "{question}\n{choices}\nAnswer:",
            "ja": "{question}\n{choices}\n答え:",
            "pt": "{question}\n{choices}\nResposta:",
            "es": "{question}\n{choices}\nRespuesta:",
            "fr": "{question}\n{choices}\nRéponse:",
            "de": "{question}\n{choices}\nAatwort:",
        },
        "mmlu": {
            "en": "{question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "{question}.\n選択肢: \n{choices}.\n答え:",
            "pt": "{question}.\nResposta: \n{choices}.\nResposta:",
            "es": "{question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "{question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "{question}.\nAatworten: {choices}.\nAatwort:",
        },
    },
    "with_context.no_intro": {
        "helm": {
            "en": "Context: {context}\nQuestion: {question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "テキスト: {context}\n質問: {question}.\n選択肢: \n{choices}.\n答え:",
            "pt": "Contexto: {context}\nPergunta: {question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Contexto: {context}\nPregunta: {question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Contexte: {context}\nQuestion: {question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Zusammenhang: {context}\nFrage: {question}.\nAatworten: {choices}.\nAatwort:",
        },
        "mmlu": {
            "en": "{context}\n{question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "{context}\n{question}.\n選択肢: \n{choices}.\n答え:",
            "pt": "{context}\n{question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "{context}\n{question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "{context}\n{question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "{context}\n{question}.\nAatworten: \n{choices}.\nAatwort:",
        },
        "lm_eval_harness": {
            "en": "{context}\n{question}\n{choices}\nAnswer:",
            "ja": "{context}\n{question}\n{choices}\n答え:",
            "pt": "{context}\n{question}\n{choices}\nResposta:",
            "es": "{context}\n{question}\n{choices}\nRespuesta:",
            "fr": "{context}\n{question}.\n{choices}\nRéponse:",
            "de": "{context}\n{question}.\n{choices}\nAatwort:",
        },
    },
    "with_context.with_topic": {
        "mmlu": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n{context}\n{question}.\n答え: {choices}.\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n{context}\n{question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n{context}\n{question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n{context}\n{question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n{context}\n{question}.\nAatworten: {choices}.\nAatwort:",
        },
        "helm": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}.\nAnswers: \n{choices}.\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n\nテキスト: {context}\n質問: {question}.\n答え: {choices}.\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n\nContexto: {context}\nPergunta: {question}.\nRespostas: \n{choices}.\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n\nContexto: {context}\nPregunta: {question}.\nRespuestas: \n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n\nContexte: {context}\nQuestion: {question}.\nRéponses \n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n\nZusammenhang: {context}\nFrage: {question}.\nAatworten: {choices}.\nAatwort:",
        },
        "lm_eval_harness": {
            "en": "The following are multiple choice questions (with answers) about {topic}.\n\n{context}\n{question}\n{choices}\nAnswer:",
            "ja": "次は {topic}に関する選択式の問題です。\n\n{context}\n{question}.\n{choices}\n答え:",
            "pt": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {topic}.\n\n{context}\n{question}.\n{choices}\nResposta:",
            "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {topic}.\n\n{context}\n{question}.\n{choices}.\nRespuesta:",
            "fr": "Ce qui suit sont des questions à choix multiples (avec réponses) concernant {topic}.\n\n{context}\n{question}.\n{choices}.\nRéponse:",
            "de": "Das folgende sind mehrfache auswahlfragen (mit antworten) bezueglich {topic}.\n\n{context}\n{question}.\n{choices}.\nAatwort:",
        },
    },
}
template_handles = []

for template_type, template_type_groups in templates.items():
    for benchmark_name, template_groups in template_type_groups.items():
        for language, input_format in template_groups.items():
            template = MultipleChoiceTemplate(
                input_format=input_format,
                target_field="answer",
                choices_separator="\n",
                target_choice_format=" {choice_numeral}"
                if "lm_eval_harness" in benchmark_name
                else "{choice_numeral}",
                postprocessors=["processors.first_character"],
            )
            template_handle = f"templates.qa.multiple_choice.{template_type}.{language}.{benchmark_name}".replace(
                ".en", ""
            )
            add_to_catalog(
                template,
                template_handle,
                overwrite=True,
            )

            template_handles.append(
                {
                    "handle": template_handle,
                    "template_type": template_type,
                    "language": language,
                }
            )

template_handles = pd.DataFrame(template_handles)
for template_type in template_handles.template_type.unique():
    for lang in template_handles.language.unique():
        template_handle_list = template_handles.query(
            f'language=="{lang}" and template_type=="{template_type}"'
        ).handle.tolist()

        cur_handle = ".".join(template_handle_list[0].split(".")[:-1]) + ".all"

        add_to_catalog(
            artifact=TemplatesList(
                template_handle_list,
            ),
            name=cur_handle,
            overwrite=True,
        )


output_format = "{answer}"

# MMLU (original)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.mmlu",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_context.with_topic.mmlu",
    overwrite=True,
)

# HELM

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.helm",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_context.with_topic.helm",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        input_format="Question: {question}\n{choices}\n",
        target_prefix="Answer: ",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.helm",
    overwrite=True,
)

# lm_eval_harness

input_format = "Question: {question}\nChoices:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.lm_eval_harness",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question based on the Choices (choose from {numerals}).",
        input_format="Question:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
    ),
    "templates.qa.multiple_choice.title",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question based on the Choices (choose from {numerals}).",
        input_format="Question:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        target_choice_format="{choice_numeral}. {choice_text}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.match_closest_option",
        ],
    ),
    "templates.qa.multiple_choice.match",
    overwrite=True,
)


input_format = "Context: {context}\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_context.lm_eval_harness",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question from one of the Choices (choose from {numerals}) based on the {context_type}.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
        title_fields=["context_type"],
    ),
    "templates.qa.multiple_choice.with_context.title",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question from one of the Choices (choose from {numerals}) based on the {context_type}.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        target_choice_format="{choice_numeral}. {choice_text}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.match_closest_option",
        ],
        title_fields=["context_type"],
    ),
    "templates.qa.multiple_choice.with_context.match",
    overwrite=True,
)

# fm_eval

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.fm_eval",
    overwrite=True,
)

input_format = "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_context.with_topic.fm_eval",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question about {topic} from one of the Choices (choose from {numerals}) based on the {context_type}.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
        title_fields=["context_type"],
    ),
    "templates.qa.multiple_choice.with_context.with_topic.title",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question about {topic} from one of the Choices (choose from {numerals}) based on the {context_type}.",
        input_format="{context_type}:\n{context}\nQuestion:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        target_choice_format="{choice_numeral}. {choice_text}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.match_closest_option",
        ],
        title_fields=["context_type"],
    ),
    "templates.qa.multiple_choice.with_context.with_topic.match",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question about {topic} from one of the Choices (choose from {numerals}).",
        input_format="Question:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
    ),
    "templates.qa.multiple_choice.with_topic.title",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Answer the multiple choice Question about {topic} from one of the Choices (choose from {numerals}).",
        input_format="Question:\n{question}\nChoices:\n{choices}",
        target_prefix="Answer:\n",
        target_field="answer",
        choices_separator="\n",
        target_choice_format="{choice_numeral}. {choice_text}",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.match_closest_option",
        ],
    ),
    "templates.qa.multiple_choice.with_topic.match",
    overwrite=True,
)


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


add_to_catalog(
    TemplatesList(
        remove_duplicates(
            [
                "templates.qa.multiple_choice.with_context.lm_eval_harness",
                *[
                    i.__id__
                    for i in fetch_artifact(
                        "templates.qa.multiple_choice.with_context.no_intro.all"
                    )[0].items
                ],
                "templates.qa.multiple_choice.with_context.title",
                "templates.qa.multiple_choice.with_context.match",
            ]
        )
    ),
    "templates.qa.multiple_choice.with_context.all",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        remove_duplicates(
            [
                "templates.qa.multiple_choice.with_context.with_topic.fm_eval",
                "templates.qa.multiple_choice.with_context.with_topic.mmlu",
                "templates.qa.multiple_choice.with_context.with_topic.helm",
                *[
                    i.__id__
                    for i in fetch_artifact(
                        "templates.qa.multiple_choice.with_context.with_topic.all"
                    )[0].items
                ],
                "templates.qa.multiple_choice.with_context.with_topic.title",
                "templates.qa.multiple_choice.with_context.with_topic.match",
            ]
        )
    ),
    "templates.qa.multiple_choice.with_context.with_topic.all",
    overwrite=True,
)


add_to_catalog(
    TemplatesList(
        remove_duplicates(
            [
                "templates.qa.multiple_choice.with_topic.fm_eval",
                "templates.qa.multiple_choice.with_topic.mmlu",
                "templates.qa.multiple_choice.with_topic.helm",
                *[
                    i.__id__
                    for i in fetch_artifact(
                        "templates.qa.multiple_choice.with_topic.all"
                    )[0].items
                ],
                "templates.qa.multiple_choice.with_topic.title",
                "templates.qa.multiple_choice.with_topic.match",
            ]
        )
    ),
    "templates.qa.multiple_choice.with_topic.all",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.with_topic.mmlu",
            "templates.qa.multiple_choice.with_topic.helm",
            "templates.qa.multiple_choice.with_topic.lm_eval_harness",
        ]
    ),
    "templates.qa.multiple_choice.with_topic.blue_bench",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.open.helm",
            "templates.qa.multiple_choice.open.lm_eval_harness",
            "templates.qa.multiple_choice.open.mmlu",
        ]
    ),
    "templates.qa.multiple_choice.open.blue_bench",
    overwrite=True,
)
add_to_catalog(
    TemplatesList(
        [
            "templates.qa.multiple_choice.with_context.lm_eval_harness",
            "templates.qa.multiple_choice.with_context.no_intro.helm",
            "templates.qa.multiple_choice.with_context.no_intro.mmlu",
        ]
    ),
    "templates.qa.multiple_choice.with_context.blue_bench",
    overwrite=True,
)
