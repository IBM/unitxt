from typing import Any, Dict, List, Optional

from .operator import InstanceOperator


class IobExtractor(InstanceOperator):
    """A class designed to extract entities from sequences of text using the Inside-Outside-Beginning (IOB) tagging convention. It identifies entities based on IOB tags and categorizes them into predefined labels such as Person, Organization, and Location.

    Attributes:
        labels (List[str]): A list of entity type labels, e.g., ["Person", "Organization", "Location"].
        begin_labels (List[str]): A list of labels indicating the beginning of an entity, e.g., ["B-PER", "B-ORG", "B-LOC"].
        inside_labels (List[str]): A list of labels indicating the continuation of an entity, e.g., ["I-PER", "I-ORG", "I-LOC"].
        outside_label (str): The label indicating tokens outside of any entity, typically "O".

    The extraction process identifies spans of text corresponding to entities and labels them according to their entity type. Each span is annotated with a start and end character offset, the entity text, and the corresponding label.

    Example of instantiation and usage:
    ```python
    operator = IobExtractor(
        labels=["Person", "Organization", "Location"],
        begin_labels=["B-PER", "B-ORG", "B-LOC"],
        inside_labels=["I-PER", "I-ORG", "I-LOC"],
        outside_label="O",
    )

    instance = {
        "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
        "tokens": ["John", "Doe", "works", "at", "OpenAI"]
    }
    processed_instance = operator.process(instance)
    print(processed_instance["spans"])
    # Output: [{'start': 0, 'end': 8, 'text': 'John Doe', 'label': 'Person'}, ...]
    ```

    For more details on the IOB tagging convention, see: https://en.wikipedia.org/wiki/Inside-outside-beginning_(tagging)

    """

    labels: List[str]
    begin_labels: List[str]
    inside_labels: List[str]
    outside_label: int

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        labels = instance["labels"]
        tokens = instance["tokens"]
        text = instance["text"]

        spans = []
        current_pos = 0
        end_pos = 0

        for label, token in zip(labels, tokens):
            token_pos = text.find(token, current_pos)
            if token_pos == -1:
                raise ValueError(
                    f"Token '{token}' not found in text '{text}' starting from position {current_pos}"
                )

            end_pos = token_pos + len(token)

            if label in self.begin_labels:
                span = {
                    "start": token_pos,
                    "label": self.labels[self.begin_labels.index(label)],
                    "end": end_pos,
                }
                spans.append(span)
            elif label in self.inside_labels and spans:
                spans[-1]["end"] = end_pos

            current_pos = end_pos

        for span in spans:
            span["text"] = text[span["start"] : span["end"]]

        instance["spans"] = spans
        return instance
