import evaluate
from datasets import load_dataset
from unitxt.text_utils import print_dict

dataset = load_dataset(
    "unitxt/data",
    "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
    trust_remote_code=True,
)

print_dict(dataset["train"][0])

metric = evaluate.load("unitxt/metric")

results = metric.compute(
    predictions=["entailment" for t in dataset["test"]], references=dataset["test"]
)

print_dict(results[0])
