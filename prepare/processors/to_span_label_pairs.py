from unitxt import add_to_catalog
from unitxt.logging_utils import get_logger
from unitxt.operator import SequentialOperator
from unitxt.processors import (
    DictOfListsToPairs,
    ListToEmptyEntitiesTuples,
    LoadJson,
    RegexParser,
)

logger = get_logger()

# parse string like "1:hlle, 2:world" list of tuples using regex
regex = r"\s*((?:[^,:\\]|\\.)+?)\s*:\s*((?:[^,:\\]|\\.)+?)\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex, field="TBD")

example = "h \\:r:hello, t7 ?t : world"

logger.info(parser.process_value(example))
assert parser.process_value(example) == [("h \\:r", "hello"), ("t7 ?t", "world")]

add_to_catalog(
    SequentialOperator(
        steps=[
            RegexParser(regex=regex, field="prediction", process_every_value=False),
            RegexParser(regex=regex, field="references", process_every_value=True),
        ]
    ),
    "processors.to_span_label_pairs",
    overwrite=True,
)

regex = r"\s*((?:\\.|[^,])+?)\s*(?:,|$)()"
termination_regex = r"^\s*None\s*$"

# test regext parser
parser = RegexParser(regex=regex, termination_regex=termination_regex, field="TBD")
example = "h \\:r, t7 ?t"

logger.info(parser.process_value(example))
assert parser.process_value(example) == [("h \\:r", ""), ("t7 ?t", "")]

example = "None"
logger.info(parser.process_value(example))
assert parser.process_value(example) == []

add_to_catalog(
    SequentialOperator(
        steps=[
            RegexParser(
                regex=regex,
                termination_regex=termination_regex,
                field="prediction",
                process_every_value=False,
            ),
            RegexParser(
                regex=regex,
                termination_regex=termination_regex,
                field="references",
                process_every_value=True,
            ),
        ]
    ),
    "processors.to_span_label_pairs_surface_only",
    overwrite=True,
)

parser = LoadJson(field="TBD")
operator = DictOfListsToPairs(position_key_before_value=False, field="TBD")

example = '{"PER":["david", "james"]}'
parsed = parser.process_value(example)
logger.info(parsed)
converted = operator.process_value(parsed)
logger.info(converted)
assert converted == [("david", "PER"), ("james", "PER")]
add_to_catalog(
    SequentialOperator(
        steps=[
            LoadJson(field="prediction", process_every_value=False),
            LoadJson(field="references", process_every_value=True),
        ]
    ),
    "processors.load_json",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            LoadJson(field="prediction", process_every_value=False),
        ]
    ),
    "processors.load_json_from_predictions",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            DictOfListsToPairs(
                position_key_before_value=False,
                field="prediction",
                process_every_value=False,
            ),
            DictOfListsToPairs(
                position_key_before_value=False,
                field="references",
                process_every_value=True,
            ),
        ]
    ),
    "processors.dict_of_lists_to_value_key_pairs",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ListToEmptyEntitiesTuples(field="prediction", process_every_value=False),
            ListToEmptyEntitiesTuples(field="references", process_every_value=True),
        ]
    ),
    "processors.list_to_empty_entity_tuples",
    overwrite=True,
)
