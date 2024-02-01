import unittest

from src.unitxt.serializers import (
    IndexedRowMajorTableSerializer,
    MarkdownTableSerializer,
)
from src.unitxt.test_utils.operators import (
    check_operator,
)


class TestSerializers(unittest.TestCase):
    def test_markdown_tableserializer(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                }
            }
        ]

        serialized_str = "|name|age|\n|---|---|\n|Alex|26|\n|Raj|34|\n|Donald|39|"

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                },
                "context": serialized_str,
            }
        ]

        check_operator(
            operator=MarkdownTableSerializer(field_to_field={"table": "context"}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_indexedrowmajor_tableserializer(self):
        inputs = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                }
            }
        ]

        serialized_str = (
            "col : name | age row 1 : Alex | 26 row 2 : Raj | 34 row 3 : Donald | 39"
        )

        targets = [
            {
                "table": {
                    "header": ["name", "age"],
                    "rows": [["Alex", "26"], ["Raj", "34"], ["Donald", "39"]],
                },
                "context": serialized_str,
            }
        ]

        check_operator(
            operator=IndexedRowMajorTableSerializer(
                field_to_field={"table": "context"}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
