import csv
import io
from abc import abstractmethod
from typing import Any, Dict, Union

from .operators import InstanceFieldOperator
from .type_utils import isoftype
from .types import Dialog, Image, Number, Table, Text


class Serializer(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]) -> str:
        return self.serialize(value, instance)

    @abstractmethod
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        pass


class DefaultSerializer(Serializer):
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        return str(value)


class DefaultListSerializer(Serializer):
    def serialize(self, value: Any, instance: Dict[str, Any]) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)


class DialogSerializer(Serializer):
    def serialize(self, value: Dialog, instance: Dict[str, Any]) -> str:
        # Convert the Dialog into a string representation, typically combining roles and content
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in value)


class NumberSerializer(Serializer):
    def serialize(self, value: Number, instance: Dict[str, Any]) -> str:
        # Check if the value is an integer or a float
        if isinstance(value, int):
            return str(value)
        # For floats, format to one decimal place
        if isinstance(value, float):
            return f"{value:.1f}"
        raise ValueError("Unsupported type for NumberSerializer")


class NumberQuantizingSerializer(NumberSerializer):
    quantum: Union[float, int] = 0.1

    def serialize(self, value: Number, instance: Dict[str, Any]) -> str:
        if isoftype(value, Number):
            quantized_value = round(value / self.quantum) / (1 / self.quantum)
            if isinstance(self.quantum, int):
                quantized_value = int(quantized_value)
            return str(quantized_value)
        raise ValueError("Unsupported type for NumberSerializer")


class TableSerializer(Serializer):
    def serialize(self, value: Table, instance: Dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")

        # Write the header and rows to the CSV writer
        writer.writerow(value["header"])
        writer.writerows(value["rows"])

        # Retrieve the CSV string
        return output.getvalue().strip()


class ImageSerializer(Serializer):
    def serialize(self, value: Image, instance: Dict[str, Any]) -> str:
        if "media" not in instance:
            instance["media"] = {}
        if "images" not in instance["media"]:
            instance["media"]["images"] = []
        idx = len(instance["media"]["images"])
        instance["media"]["images"].append(value)
        return f'<img src="media/images/{idx}">'


class DynamicSerializer(Serializer):
    image: Serializer = ImageSerializer()
    number: Serializer = DefaultSerializer()
    table: Serializer = TableSerializer()
    dialog: Serializer = DialogSerializer()
    text: Serializer = DefaultSerializer()
    list: Serializer = DefaultSerializer()

    def serialize(self, value: Any, instance: Dict[str, Any]) -> Any:
        if isoftype(value, Image):
            return self.image.serialize(value, instance)

        if isoftype(value, Table):
            return self.table.serialize(value, instance)

        if isoftype(value, Dialog) and len(value) > 0:
            return self.dialog.serialize(value, instance)

        if isoftype(value, Text):
            return self.text.serialize(value, instance)

        if isoftype(value, Number):
            return self.number.serialize(value, instance)

        if isinstance(value, list):
            return self.list.serialize(value, instance)

        return str(value)
