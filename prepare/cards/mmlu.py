# from src.unitxt.blocks import (
#     LoadHF,
#     SplitRandomMix,
#     AddFields,
#     TaskCard,
#     NormalizeListFields,
#     FormTask,
#     TemplatesList,
#     InputOutputTemplate,
#     MapInstanceValues
# )
# from src.unitxt.test_utils.card import test_card
#
# from src.unitxt.catalog import add_to_catalog
# from src.unitxt.templates import MultipleChoiceInputOutputTemplate, TemplatesDict
# from src.unitxt.operators import RenameFields
# from unitxt.splitters import RenameSplits
#
# sub_task = 'abstract_algebra'
#
# # numbering=tuple(str(x) for x in range(200))
# numbering = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# expected_answer = "number"  # "number_and_answer" #"number"
#
# card = TaskCard(
#     loader=LoadHF(path='cais/mmlu', name=sub_task),
#     preprocess_steps=[
#         RenameSplits({"auxiliary_train": "train"}),
#         RenameFields({'answer': 'label', 'question': 'sentence1'}),
#
#         # TODO copy logic from: MultipleChoiceInputOutputTemplate
#         # TODO delete MultipleChoiceInputOutputTemplate
#         AddGlobalFields({"numbering": numbering,
#                          "topic": sub_task.replace("_", " ")}),
#         ZiplLists(shortest=True, from_fields=["numbering", "label"], to_field="choices"),
#         Join(shortest=True, separator=". ", field="choices/*", to_field="choices_list"),
#         TakeFromList(field="choices_list", index="answer", to_field="number_and_answer"),
#         TakeFromList(field="numbering", index="answer", to_field="number"),
#         Join(shortest=True, separator=",", field="choices/*/0"),
#         Join(shortest=True, separator=",", field="choices_list", to_field="choices"),  # field_to_field
#         RenameFields({expected_answer: "label"})
#     ],
#     task=FormTask(
#         inputs=['choices', 'sentence1'],
#         outputs=["label", ],
#         metrics=['metrics.accuracy'],
#     ),
#     templates=TemplatesDict({
#         "original": InputOutputTemplate(
#             input_format="""
#                         The following are multiple choice questions (with answers) about {topic}.\n
#                         {sentence1}.\nAnswers: {choices}.\nAnswer:
#                 """.strip(),
#             output_format='{label}',
#         ),
#         "helm": InputOutputTemplate(
#             input_format="""
#                         The following are multiple choice questions (with answers) about {topic}.\n\n
#                         Question: {sentence1}.\nAnswers: {choices}.\nAnswer:
#                 """.strip(),
#             output_format='{label}',
#         ),
#         "LM_eval_harness": InputOutputTemplate(
#             input_format="""
#                         Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:
#                 """.strip(),
#             output_format='{label}',
#         ),
#         "fm-eval": InputOutputTemplate(
#             input_format="""
#                         The following are multiple choice questions (with answers) about {topic}.\n\n
#                         Question: {sentence1}.\nChoose from {labels}\nAnswers: {choices}.\nAnswer:
#                 """.strip(),
#             output_format='{label}',
#         ),
#     })
# )
# test_card(card)
# add_to_catalog(card, f'cards.mmlu.{sub_task}', overwrite=True)
