"""This section describes unitxt operators.

Operators: Building Blocks of Unitxt Processing Pipelines
==============================================================

Within the Unitxt framework, operators serve as the foundational elements used to assemble processing pipelines.
Each operator is designed to perform specific manipulations on dictionary structures within a stream.
These operators are callable entities that receive a MultiStream as input.
The output is a MultiStream, augmented with the operator's manipulations, which are then systematically applied to each instance in the stream when pulled.

Creating Custom Operators
-------------------------------
To enhance the functionality of Unitxt, users are encouraged to develop custom operators.
This can be achieved by inheriting from any of the existing operators listed below or from one of the fundamental :class:`base operators<unitxt.operator>`.
The primary task in any operator development is to implement the `process` function, which defines the unique manipulations the operator will perform.

General or Specelized Operators
--------------------------------
Some operators are specielized in specific task such as:

- :class:`loaders<unitxt.loaders>` for loading data.
- :class:`splitters<unitxt.splitters>` for fixing data splits.
- :class:`struct_data_operators<unitxt.struct_data_operators>` for structured data operators.

Other specelized operators are used by unitxt internally:

- :class:`templates<unitxt.templates>` for verbalizing data examples.
- :class:`formats<unitxt.formats>` for preparing data for models.

The rest of this section is dedicated for operators that operates on streams.

"""

from typing import (
    List,
    Literal,
    Optional,
)

import pandas as pd

from .operator import (
    MultiStream,
    MultiStreamOperator,
)
from .settings_utils import get_settings
from .stream import ListStream

settings = get_settings()


class JoinStreams(MultiStreamOperator):
    """Join multiple streams into a single stream.

    Args:
        left_stream (str): The stream that will be considered the "left" in the join operations.
        right_stream (str): The stream that will be considered the "right" in the join operations.
        how (Literal["left", "right", "inner", "outer", "cross"]): The type of join to be performed.
        on (Optional[List[str]]): Column names to join on. These must be found in both streams.
        left_on (Optional[List[str]]): Column  names to join on in the left stream.
        right_on (Optional[List[str]]): Column  names to join on in the right streasm.
        new_stream_name (str): The name of the new stream resulting from the merge.

    Examples:
       JoinStreams(left_stream = "questions", right_stream = "answers", how="inner", on="question_id", new_stream_name="question_with_answers" ) Join the 'question' and 'answer' stream based on the 'question_id' field using inner join, resulting with a new stream named "question_with_answers".
       JoinStreams(left_stream = "questions", right_stream = "answers", how="inner", on_left="question_id", on_right="question" new_stream_name="question_with_answers" ) Join the 'question' and 'answer' stream based on the 'question_id' field in the left stream and the 'question' field in the right stream, using inner join, resulting with a new stream named "question_with_answers". This is suitable when the fields have different labels across the streams.
    """

    left_stream: str
    right_stream: str
    how: Literal["left", "right", "inner", "outer", "cross"]
    on: Optional[List[str]] = None
    left_on: Optional[List[str]] = None
    right_on: Optional[List[str]] = None
    new_stream_name: str

    def merge(self, multi_stream) -> List:
        assert self.right_stream in multi_stream and self.left_stream in multi_stream
        stream_dict = dict(multi_stream.items())
        left_stream = list(stream_dict[self.left_stream])
        right_stream = list(stream_dict[self.right_stream])
        left_stream_df = pd.DataFrame(left_stream)
        right_stream_df = pd.DataFrame(right_stream)

        # Remove common col we don't join on, so we don't have unexpected column (standard behavior is to add a suffix)
        common_cols = set(left_stream_df.columns).intersection(
            set(right_stream_df.columns)
        )
        on = self.on if self.on is not None else []
        left_on = self.left_on if self.left_on is not None else []
        right_on = self.right_on if self.right_on is not None else []
        on_cols = set(on + left_on + right_on)
        col_to_remove = list(common_cols - on_cols)
        left_stream_df = left_stream_df.drop(columns=col_to_remove, errors="ignore")
        right_stream_df = right_stream_df.drop(columns=col_to_remove, errors="ignore")

        merged_df = pd.merge(
            left_stream_df,
            right_stream_df,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
        )
        return merged_df.to_dict(orient="records")

    def process(self, multi_stream: MultiStream) -> MultiStream:
        merged_records = self.merge(multi_stream)
        multi_stream[self.new_stream_name] = ListStream(instances_list=merged_records)
        return multi_stream


class DeleteSplits(MultiStreamOperator):
    """Operator which delete splits in stream.

    Attributes:
        splits (List[str]): The splits to delete from the stream.
    """

    splits: List[str]

    def process(self, multi_stream: MultiStream) -> MultiStream:
        generators = {
            key: val for key, val in multi_stream.items() if key not in self.splits
        }
        return MultiStream(generators)
