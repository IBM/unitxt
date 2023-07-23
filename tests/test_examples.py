import unittest


class TestExamples(unittest.TestCase):
    def test_example1(self):
        import examples.example1
        self.assertTrue(True)

    def test_example2(self):
        import examples.example2
        self.assertTrue(True)

    def test_example3(self):
        import examples.example3
        self.assertTrue(True)

    def test_add_metric_to_catalog(self):
        import examples.add_metric_to_catalog
        self.assertTrue(True)

    def test_add_recipe_to_catalog(self):
        import examples.add_recipe_to_catalog
        self.assertTrue(True)

    def test_example6(self):
        import examples.example6
        self.assertTrue(True)

    def test_evaluate(self):
        import examples.evaluate
        self.assertTrue(True)

    def test_load_dataset(self):
        import examples.load_dataset
        self.assertTrue(True)

    def test_example9(self):
        import examples.example9
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()