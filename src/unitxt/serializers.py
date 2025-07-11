import csv
import io
import json
from abc import abstractmethod
from typing import Any, Dict, List, Union

from .dataclass import AbstractField, Field
from .operators import InstanceFieldOperator
from .settings_utils import get_constants
from .type_utils import isoftype, to_type_string
from .types import (
    Conversation,
    Dialog,
    Document,
    Image,
    MultiDocument,
    Number,
    SQLDatabase,
    Table,
    Tool,
    ToolCall,
    Video,
)

constants = get_constants()


class Serializer(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]) -> str:
        return self.serialize(value, instance)

    @abstractmethod
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        pass


class DefaultSerializer(Serializer):
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        return str(value)


class SingleTypeSerializer(InstanceFieldOperator):
    serialized_type: object = AbstractField()

    def process_instance_value(self, value: Any, instance: Dict[str, Any]) -> str:
        if not isoftype(value, self.serialized_type):
            raise ValueError(
                f"SingleTypeSerializer for type {self.serialized_type} should get this type. got {to_type_string(value)}"
            )
        return self.serialize(value, instance)


class DefaultListSerializer(Serializer):
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)


class ListSerializer(SingleTypeSerializer):
    serialized_type = list

    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        return ", ".join(str(item) for item in value)


class DictAsJsonSerializer(SingleTypeSerializer):
    serialized_type = dict

    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        return json.dumps(value)


class DialogSerializer(SingleTypeSerializer):
    serialized_type = Dialog

    def serialize(self, value: Dialog, instance: Dict[str, Any]) -> str:
        # Convert the Dialog into a string representation, typically combining roles and content
        turns = []
        for turn in value:
            turn_str = f"{turn['role']}: "
            if "content" in turn:
                turn_str += str(turn["content"])
            if "tool_calls" in turn:
                turn_str += "\n" + json.dumps(turn["tool_calls"])
            turns.append(turn_str)
        return "\n".join(turns)


class ConversationSerializer(DialogSerializer):
    serialized_type = Conversation

    def serialize(self, value: Conversation, instance: Dict[str, Any]) -> str:
        return super().serialize(value["dialog"], instance)


class NumberSerializer(SingleTypeSerializer):
    serialized_type = Number

    def serialize(self, value: Number, instance: Dict[str, Any]) -> str:
        # Check if the value is an integer or a float
        if isinstance(value, int):
            return str(value)
        # For floats, format to one decimal place
        if isinstance(value, float):
            return f"{value:.1f}"
        raise ValueError("Unsupported type for NumberSerializer")


class NumberQuantizingSerializer(NumberSerializer):
    serialized_type = Number
    quantum: Union[float, int] = 0.1

    def serialize(self, value: Number, instance: Dict[str, Any]) -> str:
        if isoftype(value, Number):
            quantized_value = round(value / self.quantum) / (1 / self.quantum)
            if isinstance(self.quantum, int):
                quantized_value = int(quantized_value)
            return str(quantized_value)
        raise ValueError("Unsupported type for NumberSerializer")


class TableSerializer(SingleTypeSerializer):
    serialized_type = Table

    def serialize(self, value: Table, instance: Dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")

        # Write the header and rows to the CSV writer
        writer.writerow(value["header"])
        writer.writerows(value["rows"])

        # Retrieve the CSV string
        return output.getvalue().strip()


class ImageSerializer(SingleTypeSerializer):
    serialized_type = Image

    def serialize(self, value: Image, instance: Dict[str, Any]) -> str:
        if "media" not in instance:
            instance["media"] = {}
        if "images" not in instance["media"]:
            instance["media"]["images"] = []
        idx = len(instance["media"]["images"])
        instance["media"]["images"].append(
            {"image": value["image"], "format": value["format"]}
        )
        value["image"] = f"media/images/{idx}"
        return f'<{constants.image_tag} src="media/images/{idx}">'


class VideoSerializer(ImageSerializer):
    serialized_type = Video

    def serialize(self, value: Video, instance: Dict[str, Any]) -> str:
        serialized_images = []
        for image in value:
            image = super().serialize(image, instance)
            serialized_images.append(image)
        return "".join(serialized_images)


class DocumentSerializer(SingleTypeSerializer):
    serialized_type = Document

    def serialize(self, value: Document, instance: Dict[str, Any]) -> str:
        return f"# {value['title']}\n\n{value['body']}"


class MultiDocumentSerializer(DocumentSerializer):
    serialized_type = MultiDocument

    def serialize(self, value: MultiDocument, instance: Dict[str, Any]) -> str:
        documents = []
        for document in value:
            documents.append(super().serialize(document, instance))
        return "\n\n".join(documents)


class ToolsSerializer(SingleTypeSerializer):
    serialized_type = List[Tool]

    def serialize(self, value: List[Tool], instance: Dict[str, Any]) -> str:
        if "__tools__" not in instance:
            instance["__tools__"] = []
        tool = []
        for tool in value:
            instance["__tools__"].append({"type": "function", "function": tool})
        return json.dumps(instance["__tools__"], indent=4)


class ToolCallSerializer(SingleTypeSerializer):
    serialized_type = ToolCall

    def serialize(self, value: ToolCall, instance: Dict[str, Any]) -> str:
        return json.dumps(value)


class MultiTypeSerializer(Serializer):
    serializers: List[SingleTypeSerializer] = Field(
        default_factory=lambda: [
            DocumentSerializer(),
            ToolCallSerializer(),
            DialogSerializer(),
            MultiDocumentSerializer(),
            ImageSerializer(),
            VideoSerializer(),
            TableSerializer(),
            ToolsSerializer(),
            DialogSerializer(),
        ]
    )

    def verify(self):
        super().verify()
        self._verify_serializers(self.serializers)

    def _verify_serializers(self, serializers):
        if not isoftype(serializers, List[SingleTypeSerializer]):
            raise ValueError(
                "MultiTypeSerializer requires the list of serializers to be List[SingleTypeSerializer]."
            )

    def add_serializers(self, serializers: List[SingleTypeSerializer]):
        self._verify_serializers(serializers)
        self.serializers = serializers + self.serializers

    def serialize(self, value: Any, instance: Dict[str, Any]) -> Any:
        for serializer in self.serializers:
            if isoftype(value, serializer.serialized_type):
                return serializer.serialize(value, instance)

        return str(value)


class SQLDatabaseAsSchemaSerializer(SingleTypeSerializer):
    """Serializes a database schema into a string representation."""

    serialized_type = SQLDatabase

    def serialize(self, value: SQLDatabase, instance: Dict[str, Any]) -> str:
        from .text2sql_utils import get_db_connector

        connector = get_db_connector(value["db_type"])(value)
        return connector.get_table_schema()
