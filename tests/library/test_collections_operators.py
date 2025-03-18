from unitxt.collections_operators import (
    Chunk,
    Dictify,
    DuplicateByList,
    DuplicateBySubLists,
    Get,
    Slice,
    Wrap,
)
from unitxt.processors import (
    AddPrefix,
    FixWhiteSpace,
    GetSQL,
    RemoveArticles,
    RemovePunctuations,
)
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestCollectionsOperators(UnitxtTestCase):
    def test_dictify(self):
        operator = Dictify(field="tuple", with_keys=["a", "b"], to_field="dict")

        inputs = [{"tuple": (1, 2)}]

        targets = [{"tuple": (1, 2), "dict": {"a": 1, "b": 2}}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_wrap(self):
        operator = Wrap(field="x", inside="tuple")

        inputs = [{"x": 1}]

        targets = [{"x": (1,)}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

        with self.assertRaises(ValueError):
            Wrap(field="x", inside="non_existing_collection")

    def test_slice(self):
        operator = Slice(field="x", start=1, stop=3)

        inputs = [{"x": [0, 1, 2, 3, 4]}]

        targets = [{"x": [1, 2]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get(self):
        operator = Get(field="x", item=-1)

        inputs = [{"x": [0, 1, 2, 3, 4]}]

        targets = [{"x": 4}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list(self):
        operator = DuplicateByList(field="x", to_field="y")

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [
            {"y": 0, "x": [0, 1, 2]},
            {"y": 1, "x": [0, 1, 2]},
            {"y": 2, "x": [0, 1, 2]},
            {"y": 3, "x": [3]},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list_same_field(self):
        operator = DuplicateByList(field="x")

        inputs = [{"x": [0, 1]}]

        targets = [
            {"x": 0},
            {"x": 1},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list_deep_copy(self):
        operator = DuplicateByList(field="x", use_deep_copy=True)

        inputs = [{"x": [0, 1]}]

        targets = [
            {"x": 0},
            {"x": 1},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_sub_lists(self):
        operator = DuplicateBySubLists(field="x")

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [{"x": [0]}, {"x": [0, 1]}, {"x": [0, 1, 2]}, {"x": [3]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_sub_lists_with_deep_copy(self):
        operator = DuplicateBySubLists(field="x", use_deep_copy=True)

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [{"x": [0]}, {"x": [0, 1]}, {"x": [0, 1, 2]}, {"x": [3]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_chunk(self):
        operator = Chunk(field="x", size=2)
        inputs = [{"x": [0, 1, 2]}, {"x": [0, 1]}, {"x": [3]}]
        targets = [{"x": [[0, 1], [2]]}, {"x": [[0, 1]]}, {"x": [[3]]}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_add_prefix(self):
        operator = AddPrefix(field="text", prefix="Hello ")
        inputs = [{"text": "World"}, {"text": "Hello there"}, {"text": " Hello again"}]
        targets = [
            {"text": "Hello World"},
            {"text": "Hello there"},
            {"text": "Hello again"},
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_simple_query(self):
        operator = GetSQL(field="text")
        inputs = [{"text": "SELECT * FROM table;"}]
        targets = [{"text": "SELECT * FROM table"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_code_block(self):
        operator = GetSQL(field="text")
        inputs = [{"text": "```SELECT id FROM table```"}]
        targets = [{"text": "SELECT id FROM table"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_no_query(self):
        operator = GetSQL(field="text")
        inputs = [{"text": "No SQL query here"}]
        targets = [{"text": "No query found in generation"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_complex_query(self):
        operator = GetSQL(field="text")
        inputs = [
            {
                "text": "```SELECT column1, column2, column3\n"
                "FROM table_name\n"
                "WHERE condition1 = 'value1'\n"
                "   OR condition2 BETWEEN 'value2' AND 'value3'\n"
                "   AND condition3 IN ('value4', 'value5', 'value6')\n"
                "ORDER BY column1 ASC, column2 DESC\n"
                "LIMIT 10 OFFSET 5;```"
            }
        ]
        targets = [
            {
                "text": "SELECT column1, column2, column3\n"
                "FROM table_name\n"
                "WHERE condition1 = 'value1'\n"
                "   OR condition2 BETWEEN 'value2' AND 'value3'\n"
                "   AND condition3 IN ('value4', 'value5', 'value6')\n"
                "ORDER BY column1 ASC, column2 DESC\n"
                "LIMIT 10 OFFSET 5"
            }
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_multiple_queries(self):
        operator = GetSQL(field="text")
        inputs = [{"text": "SELECT * FROM table1; SELECT * FROM table2;"}]
        targets = [{"text": "SELECT * FROM table1"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_no_semicolon(self):
        operator = GetSQL(field="text")
        inputs = [{"text": "SELECT * FROM table"}]
        targets = [{"text": "SELECT * FROM table"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_multiple_selects(self):
        operator = GetSQL(field="text")
        inputs = [
            {
                "text": "SELECT column1 FROM table1; \n Some text in the middle \n SELECT column2 FROM table2"
            }
        ]
        targets = [{"text": "SELECT column1 FROM table1"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get_sql_with_with_clause(self):
        operator = GetSQL(field="text")
        inputs = [
            {
                "text": "WITH regional_sales AS (SELECT region, SUM(amount) AS total_sales FROM sales_data GROUP BY region) SELECT region FROM regional_sales"
            }
        ]
        targets = [
            {
                "text": "WITH regional_sales AS (SELECT region, SUM(amount) AS total_sales FROM sales_data GROUP BY region) SELECT region FROM regional_sales"
            }
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_remove_articles_with_empty_input(self):
        operator = RemoveArticles(field="text")
        inputs = [{"text": ""}]
        targets = [{"text": ""}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_remove_articles_with_no_articles(self):
        operator = RemoveArticles(field="text")
        inputs = [{"text": "Hello world!"}]
        targets = [{"text": "Hello world!"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_remove_punctuations(self):
        operator = RemovePunctuations(field="text")
        inputs = [
            {"text": "Hello, world!"},
            {"text": "This is a sentence with punctuation: .,;!?"},
            {"text": "No punctuation here"},
        ]
        targets = [
            {"text": "Hello world"},
            {"text": "This is a sentence with punctuation "},
            {"text": "No punctuation here"},
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_remove_punctuations_with_empty_input(self):
        operator = RemovePunctuations(field="text")
        inputs = [{"text": ""}]
        targets = [{"text": ""}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_remove_punctuations_with_only_punctuations(self):
        operator = RemovePunctuations(field="text")
        inputs = [{"text": ".,;!?"}]
        targets = [{"text": ""}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_fix_white_space(self):
        operator = FixWhiteSpace(field="text")
        inputs = [
            {"text": "  This is a   test  "},
            {"text": "NoExtraSpacesHere"},
            {"text": "   "},
        ]
        targets = [
            {"text": "This is a test"},
            {"text": "NoExtraSpacesHere"},
            {"text": ""},
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_fix_white_space_with_empty_input(self):
        operator = FixWhiteSpace(field="text")
        inputs = [{"text": ""}]
        targets = [{"text": ""}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_fix_white_space_with_newline_and_tabs(self):
        operator = FixWhiteSpace(field="text")
        inputs = [{"text": "  \tThis is a\n test with \t\nspaces."}]
        targets = [{"text": "This is a test with spaces."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
