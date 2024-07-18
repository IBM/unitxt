from dataclasses import dataclass

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard, TemplatesDict
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, ListFieldValues, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

from prepare.tasks.rag.rag_task import (
    TaskRagCorporaConstants,
    TaskRagCorporaInputConstants,
)


@dataclass(frozen=True)
class ClapNqDocumentsConstants:
    # Raw_data
    RAW_FILE_URL: str = "https://media.githubusercontent.com/media/primeqa/clapnq/main/retrieval/passages.tsv"

    # Fields
    ID: str = "id"
    TEXT: str = "text"
    TITLE: str = "title"

    ARTIFACT_NAME: str = "cards.rag.documents.clap_nq.en"


card = TaskCard(
    loader=LoadCSV(
        sep="\t",
        files={"test": ClapNqDocumentsConstants.RAW_FILE_URL},
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                ClapNqDocumentsConstants.ID: TaskRagCorporaInputConstants.DOCUMENT_ID,
                ClapNqDocumentsConstants.TITLE: TaskRagCorporaInputConstants.TITLE,
            },
        ),
        ListFieldValues(
            fields=[ClapNqDocumentsConstants.TEXT],
            to_field=TaskRagCorporaInputConstants.PASSAGES,
        ),
        Set(
            fields={
                TaskRagCorporaInputConstants.METADATA_FIELD: "",
            }
        ),
    ],
    task=TaskRagCorporaConstants.TASKS_RAG_CORPORA,
    templates=TemplatesDict(
        {
            "empty": InputOutputTemplate(
                input_format="",
                output_format="",
            ),
        }
    ),
)

# Not strict because Rouge does not return 1.0 even if predictions are equal to references, when txt is only "-'"
# Check only one language to speed up process
test_card(
    card,
    strict=False,
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)

add_to_catalog(card, ClapNqDocumentsConstants.ARTIFACT_NAME, overwrite=True)
