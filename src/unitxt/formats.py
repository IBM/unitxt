import json
import re
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from .dataclass import OptionalField
from .operator import InstanceOperator
from .type_utils import isoftype


class Format(InstanceOperator):
    pass


def apply_capital_new_line_notation(text: str) -> str:
    r"""Transforms a given string by applying the Capital New Line Notation.

    The Capital New Line Notation (\N) is designed to manage newline behavior in a string efficiently.
    This custom notation aims to consolidate multiple newline characters (\n) into a single newline under
    specific conditions, with tailored handling based on whether there's preceding text. The function
    distinguishes between two primary scenarios:

    1. If there's text (referred to as a prefix) followed by any number of \n characters and then one or
    more \N, the entire sequence is replaced with a single \n. This effectively simplifies multiple
    newlines and notation characters into a single newline when there's preceding text.
    2. If the string starts with \n characters followed by \N without any text before this sequence, or if
    \N is at the very beginning of the string, the sequence is completely removed. This case is
    applicable when the notation should not introduce any newlines due to the absence of preceding text.

    Args:
        text (str): The input string to be transformed, potentially containing the Capital New Line Notation
                        (\N) mixed with actual newline characters (\n).

    Returns:
        str: The string after applying the Capital New Line Notation rules, which either consolidates multiple
            newlines and notation characters into a single newline when text precedes them, or removes the
            notation and any preceding newlines entirely if no text is present before the notation.

    Examples:
        >>> apply_capital_new_line_notation("Hello World\\n\\n\N")
        'Hello World\\n'

        >>> apply_capital_new_line_notation("\\n\\n\NGoodbye World")
        'Goodbye World'

        >>> apply_capital_new_line_notation("\N")
        ''
    """
    # If sequence of \N or \n that ends with \N has no characters before delete it
    text = re.sub(r"^(?:\n|\\N)*\\N", "", text)
    # Replace every sequence of \N or \n that ends with \N with \n
    return re.sub(r"[\n(\\N)]*(\\N)+", r"\n", text)


class BaseFormat(Format):
    demos_field: str = "demos"

    @staticmethod
    def _retrieve_field_and_pop_from_instance(
        instance, field_name, do_pop: bool = True
    ) -> str:
        if field_name is not None and field_name in instance:
            field_value = instance[field_name]
            if do_pop:
                instance.pop(field_name)
            assert (
                field_value is not None
            ), f"Value in field '{field_name}' should not be none. Received instance: {instance}"
            return field_value
        return ""


class SystemFormat(BaseFormat):
    r"""Generates the whole input to the model, from constant strings that are given as args, and from values found in specified fields of the instance.

    Important: formats can use '\N' notations that means new-line if no new-line before and no empty string before.

    SystemFormat expects the input instance to contain:
    1. A field named "system_prompt" whose value is a string (potentially empty) that delivers a task-independent opening text.
    2. A field named "source" whose value is a string verbalizing the original values in the instance (as read
    from the source dataset), in the context of the underlying task.
    3. A field named "instruction" that contains a (non-None) string.
    4. A field named with the value in arg 'demos_field', containing a list of dicts, each dict with fields "source"
    and "target", representing a single demo.
    5. A field named "target_prefix" that contains a string to prefix the target in each demo, and to end the whole generated prompt

    SystemFormat formats the above fields into a single string to be inputted to the model. This string overwrites
    field "source" of the instance. Formatting is driven by two args: 'demo_format' and 'model_input_format'.
    SystemFormat also pops fields "system_prompt", "instruction", "target_prefix",  and the field containing the demos out from the input instance.

    Args:
        demos_field (str): the name of the field that contains the demos, being a list of dicts, each with "source" and "target" keys
        demo_format (str): formatting string for a single demo, combining fields "source" and "target"
        model_input_format (str) overall product format, combining instruction and source (as read from fields "instruction"
        and "source" of the input instance), together with demos (as formatted into one string)
        format_args: Dict[str,str]: additional format args to be used when formatting the different format strings

    Example:
        when input instance:

        .. code-block::

            {
                "source": "1+1",
                "target": "2",
                "instruction": "Solve the math exercises.",
                "demos": [{"source": "1+2", "target": "3"}, {"source": "4-2", "target": "2"}]
            }

        is processed by

        .. code-block::

            system_format = SystemFormat(
                demos_field="demos",
                demo_format="Input: {source}\nOutput: {target}\n\n",
                model_input_format="Instruction: {instruction}\n\n{demos}Input: {source}\nOutput: ",
            )

        the resulting instance is:

        .. code-block::

            {
                "target": "2",
                "source": "Instruction: Solve the math exercises.\n\nInput: 1+2\nOutput: 3\n\nInput: 4-2\nOutput: 2\n\nInput: 1+1\nOutput: ",
            }

    """

    demo_format: str = "{source}\\N{target_prefix}{target}\n\n"  #  example: "User: {source}\nAgent: {target}\n\n"
    model_input_format: str = (
        "{system_prompt}\\N{instruction}\\N{demos}{source}\\N{target_prefix}"
    )
    format_args: Dict[str, str] = OptionalField(default_factory=dict)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        assert (
            "source" in instance
        ), f"field 'source' is expected to be in the input instance. Received instance: {instance}"
        source = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="source"
        )

        instruction = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="instruction"
        )
        target_prefix = self._retrieve_field_and_pop_from_instance(
            instance=instance,
            field_name="target_prefix",
        )
        system_prompt = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="system_prompt"
        )

        demo_instances = []
        if self.demos_field is not None and self.demos_field in instance:
            demos = instance[self.demos_field]
            assert (
                demos is not None and isoftype(demos, List[Dict[str, Any]])
            ), f"A list of dict-s is expected in field '{self.demos_field}'. Received instance: {instance}"
            demo_instances = demos
            # instance.pop(self.demos_field)

        demos_string = ""
        for demo_instance in demo_instances:
            demo_source = self._retrieve_field_and_pop_from_instance(
                instance=demo_instance, field_name="source", do_pop=False
            )
            demo_target = self._retrieve_field_and_pop_from_instance(
                instance=demo_instance, field_name="target", do_pop=False
            )
            demo_target_prefix = self._retrieve_field_and_pop_from_instance(
                instance=demo_instance, field_name="target_prefix", do_pop=False
            )

            demo_str = self.demo_format.format(
                target_prefix=demo_target_prefix,
                source=demo_source,
                target=demo_target,
                instruction=instruction,
                **self.format_args,
            )
            demos_string += demo_str

        output = self.model_input_format.format(
            system_prompt=system_prompt,
            instruction=instruction,
            demos=demos_string,
            source=source,
            target_prefix=target_prefix,
            **self.format_args,
        )
        output = apply_capital_new_line_notation(output)
        instance["source"] = output
        return instance


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlContent(TypedDict):
    type: Literal["image_url"]
    image_url: Dict[Literal["url"], str]


class ImageFileContent(TypedDict):
    type: Literal["image_file"]
    image_file: Dict[Literal["file_id"], str]


Content = Union[TextContent, ImageUrlContent, ImageFileContent]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Content]]


class OpenAIFormat(BaseFormat):
    r"""Formats output for LLM APIs using OpenAI's chat schema.

    Many API services use OpenAI's chat format as a standard for conversational models.
    `OpenAIFormat` prepares the output in this API-compatible format, converting input
    instances into OpenAI's structured chat format, which supports both text and
    multimedia elements, like images.

    The formatted output can be easily converted to a dictionary using `json.loads()`
    to make it ready for direct use with OpenAI's API.

    Example:
        Given an input instance:

        .. code-block:: python

            {
                "source": "<img src='https://example.com/image1.jpg'>What's in this image?",
                "target": "A dog",
                "instruction": "Help the user.",
            },

        When processed by:

        .. code-block:: python

            system_format = OpenAIFormat()

        The resulting formatted output is:

        .. code-block:: python

            {
                "target": "A dog",
                "source": '[{"role": "system", "content": "Help the user."}, '
                          '{"role": "user", "content": [{"type": "image_url", '
                          '"image_url": {"url": "https://example.com/image1.jpg", "detail": "low"}}, '
                          '{"type": "text", "text": "What\'s in this image?"}]}]'
            }

        This `source` field is a JSON-formatted string. To make it ready for OpenAI's API,
        you can convert it to a dictionary using `json.loads()`:

        .. code-block:: python

            import json
            formatted_instance = json.loads(formatted_output["source"])

        The resulting `formatted_instance` is now a dictionary ready for sending to the OpenAI API.
    """

    def to_content(self, text: str) -> Union[str, List[Content]]:
        # Regular expression to find <img> tags with src attribute
        img_tag_pattern = re.compile(
            r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE
        )

        # Find all matches of <img> tags and their positions
        matches = list(img_tag_pattern.finditer(text))

        # If no images are found, return the text as a plain string
        if not matches:
            return text

        contents: List[dict] = []
        last_pos = 0

        # Process each match
        for match in matches:
            start, end = match.span()
            img_url = match.group(1)

            # Add preceding text, if any
            if last_pos < start:
                contents.append({"type": "text", "text": text[last_pos:start]})

            # Add image content with a default detail level
            contents.append(
                {"type": "image_url", "image_url": {"url": img_url, "detail": "low"}}
            )

            # Update the last processed position
            last_pos = end

        # Add any remaining text after the last image
        if last_pos < len(text):
            contents.append({"type": "text", "text": text[last_pos:]})

        return contents

    def to_chat(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> List[Message]:
        assert (
            "source" in instance
        ), f"field 'source' is expected to be in the input instance. Received instance: {instance}"

        source = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="source"
        )

        instruction = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="instruction"
        )
        target_prefix = self._retrieve_field_and_pop_from_instance(
            instance=instance,
            field_name="target_prefix",
            do_pop=False,
        )
        system_prompt = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="system_prompt"
        )

        system_content = self.to_content(
            system_prompt + ("\n" if system_prompt != "" else "") + instruction,
        )

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
        ]
        demo_instances = []
        if self.demos_field is not None and self.demos_field in instance:
            demos = instance[self.demos_field]
            assert (
                demos is not None and isoftype(demos, List[Dict[str, Any]])
            ), f"A list of dict-s is expected in field '{self.demos_field}'. Received instance: {instance}"
            demo_instances = demos
            # instance.pop(self.demos_field)

        for demo_instance in demo_instances:
            user_content = self.to_content(demo_instance["source"])
            assistant_content = self.to_content(target_prefix + demo_instance["target"])
            messages.extend(
                [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]
            )

        last_user_content = self.to_content(source)

        messages.extend([{"role": "user", "content": last_user_content}])

        return messages

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        chat = self.to_chat(instance, stream_name)
        instance.pop("target_prefix", None)
        instance["source"] = json.dumps(chat)
        return instance


class HFSystemFormat(OpenAIFormat):
    r"""Formats the complete input for the model using the HuggingFace chat template of a given model.

    HFSystemFormat expects the input instance to contain:
    1. A field named "system_prompt" whose value is a string (potentially empty) that delivers a task-independent opening text.
    2. A field named "source" whose value is a string verbalizing the original values in the instance (as read
    from the source dataset), in the context of the underlying task.
    3. A field named "instruction" that contains a (non-None) string.
    4. A field named with the value in arg 'demos_field', containing a list of dicts, each dict with fields "source"
    and "target", representing a single demo.
    5. A field named "target_prefix" that contains a string to prefix the target in each demo, and to end the whole generated prompt.

    SystemFormat formats the above fields into a single string to be inputted to the model. This string overwrites
    field "source" of the instance.

    Example:
        HFSystemFormat(model_name="HuggingFaceH4/zephyr-7b-beta")

        Uses the template defined the in tokenizer_config.json of the model:

        "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",

        See more details in https://huggingface.co/docs/transformers/main/en/chat_templating

    """

    model_name: str
    _requirements_list = ["transformers", "Jinja2"]

    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        chat = self.to_chat(instance, stream_name)

        target_prefix = self._retrieve_field_and_pop_from_instance(
            instance=instance, field_name="target_prefix"
        )

        instance["source"] = (
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            + target_prefix
        )

        return instance
