from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"text": "str", "relation_type": "str"},
        outputs={
            "entity_surface_form1": "List[str]",
            "relation_type": "List[str]",
            "entity_surface_form2": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.single_relation_type",
    overwrite=True,
)

add_to_catalog(
    FormTask(
        inputs={"text": "str", "relation_types": "List[str]"},
        outputs={
            "entity_surface_form1": "List[str]",
            "relation_type": "List[str]",
            "entity_surface_form2": "List[str]",
        },
        prediction_type="List[Tuple[str,str,str]]",
        metrics=["metrics.relation_extraction"],
        augmentable_inputs=["text"],
    ),
    "tasks.relation_extraction.all_relation_types",
    overwrite=True,
)
