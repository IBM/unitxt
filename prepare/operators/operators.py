from unitxt import add_to_catalog
from unitxt.processors import RegexParser

add_to_catalog(
    RegexParser(regex=".+"),
    "operators.regex_parser",
    overwrite=True,
)
