from src.unitxt import add_to_catalog
from src.unitxt.processors import DictOfListsToPairs, LoadJson, RegexParser

# parse string like "1:hlle, 2:world" list of tuples using regex
regex = r"\s*((?:[^,:\\]|\\.)+?)\s*:\s*((?:[^,:\\]|\\.)+?)\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex)

example = "h \\:r:hello, t7 ?t : world"

print(parser.process(example))
assert parser.process(example) == [("h \\:r", "hello"), ("t7 ?t", "world")]

add_to_catalog(parser, "processors.to_span_label_pairs", overwrite=True)


regex = r"\s*((?:\\.|[^,])+?)\s*(?:,|$)()"
termination_regex = r"^\s*None\s*$"

# test regext parser
parser = RegexParser(regex=regex, termination_regex=termination_regex)
example = "h \\:r, t7 ?t"

print(parser.process(example))
assert parser.process(example) == [("h \\:r", ""), ("t7 ?t", "")]

example = "None"
print(parser.process(example))
assert parser.process(example) == []

add_to_catalog(parser, "processors.to_span_label_pairs_surface_only", overwrite=True)

parser = LoadJson()
operator = DictOfListsToPairs(position_key_before_value=False)

example = '{"PER":["david", "james"]}'
parsed = parser.process(example)
print(parsed)
converted = operator.process(parsed)
print(converted)
assert converted == [("david", "PER"), ("james", "PER")]
add_to_catalog(parser, "processors.load_json", overwrite=True)
add_to_catalog(operator, "processors.dict_of_lists_to_value_key_pairs", overwrite=True)
