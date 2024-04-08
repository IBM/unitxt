from unitxt.dataset_utils import parse

from tests.utils import UnitxtTestCase


class TestQuery(UnitxtTestCase):
    def test_query_works(self):
        query = (
            "card=cards.sst2,template_card_index=1000,demos_pool_size=100,num_demos=0"
        )
        parsed = parse(query)
        target = {
            "card": "cards.sst2",
            "template_card_index": 1000,
            "demos_pool_size": 100,
            "num_demos": 0,
        }
        self.assertDictEqual(parsed, target)

    def test_empty_query_is_ok_but_fruitless_fail(self):
        self.assertEqual({}, parse(""))

        with self.assertRaises(ValueError):
            parse(",")

    def test_missing_key_fail(self):
        with self.assertRaises(ValueError):
            parse(
                "=cards.sst2,template_card_index=1000,demos_pool_size=100,num_demos=0"
            )
        with self.assertRaises(ValueError):
            parse("cards.sst2,template_card_index=1000,demos_pool_size=100,num_demos=0")

    def test_missing_value_fail(self):
        with self.assertRaises(ValueError):
            parse(
                "cards.sst2=,template_card_index=1000,demos_pool_size=100,num_demos=0"
            )
        with self.assertRaises(ValueError):
            parse("=,template_card_index=1000,demos_pool_size=100,num_demos=0")
