from src.unitxt import add_to_catalog
from src.unitxt.metrics import BertScore, Reward, SentenceBert, TokenOverlap

metrics = [
    (
        "metrics.token_overlap",
        TokenOverlap(),
    ),
    (
        "metrics.bert_score.deberta.xlarge.mnli",
        BertScore(model_name="microsoft/deberta-xlarge-mnli"),
    ),
    (
        "metrics.sentence_bert.mpnet.base.v2",
        SentenceBert(model_name="sentence-transformers/all-mpnet-base-v2"),
    ),
    (
        "metrics.reward.deberta.v3.large.v2",
        Reward(model_name="OpenAssistant/reward-model-deberta-v3-large-v2"),
    ),
]

for metric_id, metric in metrics:
    add_to_catalog(metric, metric_id, overwrite=True)
