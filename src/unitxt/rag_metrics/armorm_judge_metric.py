import torch
from datasets import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..rag_metrics.task_based_judge_metric import TaskBasedJudgeMetric
from ..stream import MultiStream
from .rag_metrics_utils import logger


class ArmoRMMetric(TaskBasedJudgeMetric):
    _requirements_list: list[str] = ["torch", "AutoTokenizer"]
    model_name = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    num_labels = 19
    infer_batch_size = 2
    max_len = 8192

    def prepare(
        self,
    ):
        super().prepare()
        self.device = self._init_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    def _init_device(self):
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_msg = f"There are {torch.cuda.device_count()} GPUs available, using GPU {gpu_id}, name: {torch.cuda.get_device_name(gpu_id)}"
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            gpu_msg = "Using local MPS GPU"
            self.device = torch.device("mps")
        else:
            gpu_msg = "f'There are NO GPUs available.'"
            self.device = torch.device("cpu")
        logger.info(gpu_msg)

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict],
    ) -> dict:
        text_pairs = self.get_texts_pairs(task_data)
        processed_preds = self.get_scores(text_pairs)
        return [{self.main_score: s} for s in processed_preds]

    def get_scores(self, text_pairs):
        sentences_batches = [
            text_pairs[x : x + self.infer_batch_size]
            for x in range(0, len(text_pairs), self.infer_batch_size)
        ]
        scores = []
        logger.info(
            f"Inferring {len(text_pairs)} texts in {len(sentences_batches)} batches"
        )
        for sentences_batch in tqdm(sentences_batches):
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_len,
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**features)
                # there are also other possible scores to extract, as per usage example in https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
                preference_scores = output.score.cpu().float()
                batch_scores = [float(score) for score in preference_scores]
            scores.extend(batch_scores)

        return scores

    def get_templated_inputs(self, test_set):
        test_set = self.adjust_instances_to_task(test_set)
        instance_stream = MultiStream(
            {"test": [{**instance, "contexts_ids": [0]} for instance in test_set]}
        )
        return self.processor.process(instance_stream).to_dataset()["test"]["source"]

    def get_texts_pairs(self, test_set):
        templated_inputs = self.get_templated_inputs(test_set)
        return [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": premise},
                    {"role": "assistant", "content": instance["answer"]},
                ],
                tokenize=False,
            )
            for instance, premise in zip(test_set, templated_inputs)
        ]
