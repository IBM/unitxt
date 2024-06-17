from datasets import load_dataset
from unitxt.text_utils import print_dict

dataset = load_dataset(
    "unitxt/data",
    "card=cards.wnli,template_card_index=0,loader_limit=100",
    trust_remote_code=True,
)

print_dict(dataset["train"][0])
