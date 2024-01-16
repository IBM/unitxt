from src.unitxt import add_to_catalog
from src.unitxt.logging_utils import get_logger
from src.unitxt.processors import (
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

add_to_catalog(parser, "processors.to_span_label_pairs", overwrite=True)


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

add_to_catalog(parser, "processors.to_span_label_pairs_surface_only", overwrite=True)

parser = LoadJson(field="TBD")
operator = DictOfListsToPairs(position_key_before_value=False, field="TBD")

example = '{"PER":["david", "james"]}'
parsed = parser.process_value(example)
logger.info(parsed)
converted = operator.process_value(parsed)
logger.info(converted)
assert converted == [("david", "PER"), ("james", "PER")]
add_to_catalog(parser, "processors.load_json", overwrite=True)
add_to_catalog(operator, "processors.dict_of_lists_to_value_key_pairs", overwrite=True)

operator = ListToEmptyEntitiesTuples(field="TBD")
add_to_catalog(operator, "processors.list_to_empty_entity_tuples", overwrite=True)
