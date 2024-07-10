# from copy import deepcopy

# from unitxt.blocks import LoadHF, RenameFields, Set, SplitRandomMix, TaskCard
# from unitxt.catalog import add_to_catalog
# from unitxt.operators import ListFieldValues
# from unitxt.test_utils.card import test_card

# card = TaskCard(
#     loader=LoadHF(
#         path="PatronusAI/financebench",
#     ),
#     preprocess_steps=[
#         SplitRandomMix({"train": "train[10%]", "test": "train[90%]"}),
#         RenameFields(field_to_field={"answer": "answers", "evidence_text": "context"}),
#         ListFieldValues(fields=["answers"], to_field="answers"),
#         Set(fields={"context_type": "context"}),
#     ],
#     task="tasks.qa.with_context.abstractive[metrics=[metrics.rag.response_generation.correctness.bert_score.deberta_large_mnli]]",
#     templates="templates.qa.with_context.all",
# )

# # testing the card is too slow with the bert-score metric, so dropping it
# card_for_test = deepcopy(card)
# card_for_test.task.metrics = [
#     "metrics.rag.response_generation.correctness.token_overlap",
# ]

# test_card(
#     card_for_test,
#     debug=False,
#     strict=False,
#     format="formats.textual_assistant",
# )
# add_to_catalog(
#     card,
#     "cards.financebench",
#     overwrite=True,
#     catalog_path="src/unitxt/catalog",
# )
