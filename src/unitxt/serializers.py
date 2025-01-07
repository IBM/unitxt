import csv
import io
from abc import abstractmethod
from typing import Any, Dict, List, Union

from .dataclass import AbstractField, Field
from .operators import InstanceFieldOperator
from .settings_utils import get_constants
from .type_utils import isoftype, to_type_string
from .types import Dialog, Document, Image, MultiDocument, Number, Table, Video

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


class DialogSerializer(SingleTypeSerializer):
    serialized_type = Dialog

    def serialize(self, value: Dialog, instance: Dict[str, Any]) -> str:
        # Convert the Dialog into a string representation, typically combining roles and content
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in value)


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


class MultiTypeSerializer(Serializer):
    serializers: List[SingleTypeSerializer] = Field(
        default_factory=lambda: [
            DocumentSerializer(),
            MultiDocumentSerializer(),
            ImageSerializer(),
            VideoSerializer(),
            TableSerializer(),
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
