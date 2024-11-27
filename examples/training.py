from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset(
    "unitxt/data",
    "card=cards.wnli,template=templates.classification.multi_class.relation.default,max_test_instances=100",
    trust_remote_code=True,
)


def formatting(example):
    texts = []
    for i in range(len(example["source"])):
        text = example["source"][i] + example["target"][i]
        texts.append(text)
    return texts


trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset["train"],
    args=SFTConfig(output_dir="./opt-350m"),
)

trainer.train()
