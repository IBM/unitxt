from unitxt import add_to_catalog
from unitxt.logging_utils import get_logger
from unitxt.processors import (
    DictOfListsToPairs,
    ListToEmptyEntitiesTuples,
    PostProcess,
    RegexParser,
)
from unitxt.settings_utils import get_constants
from unitxt.struct_data_operators import LoadJson

constants = get_constants()
logger = get_logger()

# parse string like "1:hlle, 2:world" list of tuples using regex
regex = r"\s*((?:[^,:\\]|\\.)+?)\s*:\s*((?:[^,:\\]|\\.)+?)\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex, field="TBD")

example = "h \\:r:hello, t7 ?t : world"

logger.info(parser.process_value(example))
assert parser.process_value(example) == [("h \\:r", "hello"), ("t7 ?t", "world")]

add_to_catalog(
    PostProcess(RegexParser(regex=regex)),
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
    PostProcess(RegexParser(regex=regex, termination_regex=termination_regex)),
    "processors.to_span_label_pairs_surface_only",
    overwrite=True,
)

parser = LoadJson(field="TBD", allow_failure=True, failure_value=[])
operator = DictOfListsToPairs(position_key_before_value=False, field="TBD")

example = '{"PER":["david", "james"]}'
parsed = parser.process_value(example)
logger.info(parsed)
converted = operator.process_value(parsed)
logger.info(converted)
assert converted == [("david", "PER"), ("james", "PER")]
add_to_catalog(
    PostProcess(LoadJson(allow_failure=True, failure_value=[])),
    "processors.load_json",
    overwrite=True,
)

add_to_catalog(
    PostProcess(
        LoadJson(allow_failure=True, failure_value=[]), process_references=False
    ),
    "processors.load_json_from_predictions",
    overwrite=True,
)

add_to_catalog(
    PostProcess(
        DictOfListsToPairs(
            position_key_before_value=False,
            process_every_value=False,
        )
    ),
    "processors.dict_of_lists_to_value_key_pairs",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ListToEmptyEntitiesTuples()),
    "processors.list_to_empty_entity_tuples",
    overwrite=True,
)
