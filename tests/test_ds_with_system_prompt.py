import unittest

from datasets import load_dataset
from src import unitxt


class TestDSWithSystemPrompt(unittest.TestCase):
    def test_hf_load(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_item=0,system_prompt=xxx .{} yy",
            download_mode="force_redownload",
        )
        for example in dataset["train"][:10]["source"]:
            print(f"{example}\n\n")
