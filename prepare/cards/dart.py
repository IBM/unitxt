from unitxt.blocks import (
    AddFields,
    CopyFields,
    LoadHF,
    RenameFields,
    SerializeTriples,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="dart"),
    preprocess_steps=[
        "splitters.small_no_test",
        SerializeTriples(field_to_field=[["tripleset", "serialized_triples"]]),
        RenameFields(field_to_field={"serialized_triples": "input"}),
        CopyFields(
            field_to_field={"annotations/text/0": "output"},
        ),
        AddFields(fields={"type_of_input": "Triples", "type_of_output": "Text"}),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
    __description__=(
        "DART is a large and open-domain structured DAta Record to Text generation corpus with high-quality"
        "sentence annotations with each input being a set of entity-relation triples following a tree-structured ontology."
        "It consists of 82191 examples across different domains with each input being a semantic RDF triple set derived"
        "from data records in tables and the tree ontology of table schema, annotated with sentence description that"
        "covers all facts in the triple set."
    ),
    __tags__={
        "annotations_creators": ["crowdsourced", "machine-generated"],
        "arxiv": "2007.02871",
        "croissant": True,
        "language": "en",
        "language_creators": ["crowdsourced", "machine-generated"],
        "license": "mit",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": [
            "extended|wikitable_questions",
            "extended|wikisql",
            "extended|web_nlg",
            "extended|cleaned_e2e",
        ],
        "task_categories": "tabular-to-text",
        "task_ids": "rdf-to-text",
    },
)

test_card(card)
add_to_catalog(card, "cards.dart", overwrite=True)
