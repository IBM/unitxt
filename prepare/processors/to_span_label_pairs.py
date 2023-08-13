from src.unitxt import add_to_catalog
from src.unitxt.processors import RegexParser

# parse string like "1:hlle, 2:world" list of tuples using regex
regex = r"\s*((?:[^,:\\]|\\.)+?)\s*:\s*((?:[^,:\\]|\\.)+?)\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex)

example = "h r:hello, t7 ?t : world"

print(parser.process(example))
assert parser.process(example) == [("h r", "hello"), ("t7 ?t", "world")]

add_to_catalog(parser, "processors.to_span_label_pairs", overwrite=True)


# parse string like "1, 2" list of tuples using regex
regex = r"\s*([^,:]+?)\s*()\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex)

example = "h r, t7 ?t"

print(parser.process(example))
assert parser.process(example) == [("h r", ""), ("t7 ?t", "")]

add_to_catalog(parser, "processors.to_span_label_pairs_surface_only", overwrite=True)
