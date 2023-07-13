

from datasets import load_dataset

from src import unitxt
from src.unitxt.text_utils import print_dict

dataset = load_dataset(
    'unitxt/data',
    'card=cards.wnli,template_item=0',
)

print_dict(dataset['train'][0])