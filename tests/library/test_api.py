from src.unitxt.api import evaluate, load_dataset
from tests.utils import UnitxtTestCase


class TestAPI(UnitxtTestCase):
    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1 to 5, what is the similarity to this text An air plane is taking off.?\n",
            "target": "5.0",
            "references": ["5.0"],
            "additional_inputs": {
                "key": [
                    "text1",
                    "text2",
                    "attribute_name",
                    "min_value",
                    "max_value",
                    "attribute_value",
                ],
                "value": [
                    "A plane is taking off.",
                    "An air plane is taking off.",
                    "similarity",
                    "1",
                    "5",
                    "5.0",
                ],
            },
            "group": "unitxt",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        self.assertDictEqual(dataset["train"][0], instance)

    def test_evaluate(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertAlmostEqual(results[0]["score"]["global"]["score"], 0.026, 3)

    def test_load_dataset_with_setting_table_serializer(self):
        dataset = load_dataset(
            "card=cards.wikitq,table_serializer=serializers.table.markdown,template=templates.qa.with_context.question_first,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.squad"],
            "source": "what was the last year where this team was a part of the usl a-league?\nAnswer based on this table:\n |Year|Division|League|Regular Season|Playoffs|Open Cup|Avg. Attendance|\n|---|---|---|---|---|---|---|\n|2001|2|USL A-League|4th, Western|Quarterfinals|Did not qualify|7,169|\n|2002|2|USL A-League|2nd, Pacific|1st Round|Did not qualify|6,260|\n|2003|2|USL A-League|3rd, Pacific|Did not qualify|Did not qualify|5,871|\n|2004|2|USL A-League|1st, Western|Quarterfinals|4th Round|5,628|\n|2005|2|USL First Divis|5th|Quarterfinals|4th Round|6,028|\n|2006|2|USL First Divis|11th|Did not qualify|3rd Round|5,575|\n|2007|2|USL First Divis|2nd|Semifinals|2nd Round|6,851|\n|2008|2|USL First Divis|11th|Did not qualify|1st Round|8,567|\n|2009|2|USL First Divis|1st|Semifinals|3rd Round|9,734|\n|2010|2|USSF D-2 Pro Le|3rd, USL (3rd)|Quarterfinals|3rd Round|10,727|\n",
            "target": "2004",
            "references": ["2004"],
            "additional_inputs": {
                "key": ["context", "context_type", "question", "answer"],
                "value": [
                    "|Year|Division|League|Regular Season|Playoffs|Open Cup|Avg. Attendance|\n|---|---|---|---|---|---|---|\n|2001|2|USL A-League|4th, Western|Quarterfinals|Did not qualify|7,169|\n|2002|2|USL A-League|2nd, Pacific|1st Round|Did not qualify|6,260|\n|2003|2|USL A-League|3rd, Pacific|Did not qualify|Did not qualify|5,871|\n|2004|2|USL A-League|1st, Western|Quarterfinals|4th Round|5,628|\n|2005|2|USL First Divis|5th|Quarterfinals|4th Round|6,028|\n|2006|2|USL First Divis|11th|Did not qualify|3rd Round|5,575|\n|2007|2|USL First Divis|2nd|Semifinals|2nd Round|6,851|\n|2008|2|USL First Divis|11th|Did not qualify|1st Round|8,567|\n|2009|2|USL First Divis|1st|Semifinals|3rd Round|9,734|\n|2010|2|USSF D-2 Pro Le|3rd, USL (3rd)|Quarterfinals|3rd Round|10,727|",
                    "table",
                    "what was the last year where this team was a part of the usl a-league?",
                    "['2004']",
                ],
            },
            "group": "unitxt",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        self.assertDictEqual(dataset["train"][0], instance)
