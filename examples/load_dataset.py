from datasets import load_dataset
from unitxt.text_utils import print_dict

dataset = load_dataset(
    "unitxt/data",
    "card=cards.wnli,template_card_index=0",
)

print_dict(dataset["train"][0])
