from src.unitxt import add_to_catalog
from src.unitxt.operator import SequentialOperator
from src.unitxt.processors import DetectNoResponse

add_to_catalog(
    SequentialOperator(
        steps=[
            DetectNoResponse(field="prediction", process_every_value=False),
            DetectNoResponse(field="references", process_every_value=True),
        ]
    ),
    "processors.detect_no_response",
    overwrite=True,
)
