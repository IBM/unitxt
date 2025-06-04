import re
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from .dataclass import OptionalField
from .dict_utils import dict_get
from .error_utils import UnitxtError
from .image_operators import image_to_data_url
from .operator import InstanceOperator
from .settings_utils import get_constants
from .type_utils import isoftype
from .utils import retry_connection_with_exponential_backoff

constants = get_constants()


class Format(InstanceOperator):
    pass


class GraniteDocumentsFormat(Format):
    model: str = "ibm-granite/granite-3.1-8b-instruct"
    citations: bool = True
    length: str = "long"

    _requirements_list = ["transformers"]

    @retry_connection_with_exponential_backoff(backoff_factor=2)
    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        inputs = instance["input_fields"]
        if "question" not in inputs:
            raise UnitxtError(
                "GraniteRAGFormat works only for tasks with field: 'question'"
            )
        if "context" not in inputs and "contexts" not in inputs:
            raise UnitxtError(
                "GraniteRAGFormat works only for tasks with field: 'context' or 'contexts"
            )

        if "context" in inputs:
            texts = [inputs["context"]]
        if "contexts" in inputs:
            texts = inputs["contexts"]

        documents = []
        for text in texts:
            documents.append({"title": "", "text": text})

        question = inputs["question"]

        instance["source"] = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
            ],
            documents=documents,
            controls={"citations": self.citations, "length": self.length},
            add_generation_prompt=True,
            tokenize=False,
        )
        return instance


def apply_capital_new_line_notation(text: str) -> str:
    r"""Transforms a given string by applying the Capital New Line Notation.

    The Capital New Line Notation ``(\N)`` is designed to manage newline behavior in a string efficiently.
    This custom notation aims to consolidate multiple newline characters ``(\n)`` into a single newline under
    specific conditions, with tailored handling based on whether there's preceding text. The function
    distinguishes between two primary scenarios:

    1. If there's text (referred to as a prefix) followed by any number of ``\n`` characters and then one or
    more ``\N``, the entire sequence is replaced with a single ``\n``. This effectively simplifies multiple
    newlines and notation characters into a single newline when there's preceding text.

    2. If the string starts with ``\n`` characters followed by ``\N`` without any text before this sequence, or if
    ``\N`` is at the very beginning of the string, the sequence is completely removed. This case is
    applicable when the notation should not introduce any newlines due to the absence of preceding text.

    Args:
        text (str): The input string to be transformed, potentially containing the Capital New Line Notation ``(\N)`` mixed with actual newline characters ``(\n)``.

    Returns:
        The string after applying the Capital New Line Notation rules, which either consolidates multiple
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
    demos_field: str = constants.demos_field

    @staticmethod
    def _pop_field(instance, field_name, do_pop: bool = True) -> str:
        if field_name is not None and field_name in instance:
            field_value = instance[field_name]
            if do_pop:
                instance.pop(field_name)
            assert (
                field_value is not None
            ), f"Value in field '{field_name}' should not be none. Received instance: {instance}"
            return field_value
        return ""

    def _prepare_instance_fields(self, instance) -> Tuple[str]:
        instance_fields = {}

        for field in (
            "source",
            constants.instruction_field,
            constants.system_prompt_field,
            "target_prefix",
        ):
            instance_fields[field] = self._pop_field(instance, field)

        instance_fields["media"] = self._pop_field(instance, "media", do_pop=False)
        if not instance_fields["media"]:
            instance_fields["media"] = {"images": [], "audios": []}

        instance_fields[constants.demos_field] = []
        if self.demos_field is not None and self.demos_field in instance:
            demos = instance[self.demos_field]
            assert (
                demos is not None and isoftype(demos, List[Dict[str, Any]])
            ), f"A list of dict-s is expected in field '{self.demos_field}'. Received instance: {instance}"
            for demo_instance in demos:
                demo = {}
                for field in ["source", "target", "target_prefix"]:
                    demo[field] = self._pop_field(demo_instance, field, do_pop=False)
                instance_fields[constants.demos_field].append(demo)

        return instance_fields

    @abstractmethod
    def _format_instance_to_source(
        self,
        system_prompt: str,
        instruction: str,
        source: str,
        target_prefix: str,
        demos: List[Dict[str, Any]],
        media: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Abstract method for formatting instances in different subclasses.

        Subclasses should implement this method to define specific formatting behavior.
        """
        return ""

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance_fields = self._prepare_instance_fields(instance)
        instance["source"] = self._format_instance_to_source(**instance_fields)
        return instance


class SystemFormat(BaseFormat):
    r"""Generates the whole input to the model, from constant strings that are given as args, and from values found in specified fields of the instance.

    Important: formats can use ``'\N'`` notations that means new-line if no new-line before and no empty string before.

    SystemFormat expects the input instance to contain:
    1. A field named "system_prompt" whose value is a string (potentially empty) that delivers a task-independent opening text.
    2. A field named "source" whose value is a string verbalizing the original values in the instance (as read
    from the source dataset), in the context of the underlying task.
    3. A field named "instruction" that contains a (non-None) string.
    4. A field named with the value in arg ``'demos_field'``, containing a list of dicts, each dict with fields "source"
    and "target", representing a single demo.
    5. A field named "target_prefix" that contains a string to prefix the target in each demo, and to end the whole generated prompt

    SystemFormat formats the above fields into a single string to be input to the model. This string overwrites
    field "source" of the instance. Formatting is driven by two args: ``'demo_format'`` and ``'model_input_format'``.
    SystemFormat also pops fields "system_prompt", "instruction", "target_prefix",  and the field containing the demos out from the input instance.

    Args:
        demos_field (str): the name of the field that contains the demos, being a list of dicts, each with "source" and "target" keys
        demo_format (str): formatting string for a single demo, combining fields "source" and "target"
        model_input_format (str): overall product format, combining instruction and source (as read from fields "instruction" and "source" of the input instance), together with demos (as formatted into one string)
        format_args (Dict[str,str]): additional format args to be used when formatting the different format strings

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
                demos_field=constants.demos_field,
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

    def _format_instance_to_source(
        self,
        system_prompt: str,
        instruction: str,
        source: str,
        target_prefix: str,
        demos: List[Dict[str, Any]],
        media: Optional[Dict[str, Any]] = None,
    ) -> str:
        demos_string = ""
        for demo in demos:
            demo_str = self.demo_format.format(
                **demo,
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

        return apply_capital_new_line_notation(output)


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


class ChatAPIFormat(BaseFormat):
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

            messages = json.loads(formatted_output["source"])

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

        The resulting `messages` is now a dictionary ready for sending to the OpenAI API.
    """

    def to_content(self, text: str, media: Dict[str, Any]) -> Union[str, List[Content]]:
        # Regular expression to find <img> tags with src attribute
        img_tag_pattern = re.compile(
            r"<" + f"{constants.image_tag}" + r'\s+[^>]*src=["\']([^"\']+)["\'][^>]*>',
            re.IGNORECASE,
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
            if img_url.startswith("media/"):
                image = dict_get(media, img_url[6:])
                data_url = image_to_data_url(image)
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "low"},
                    }
                )
            else:
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url, "detail": "low"},
                    }
                )

            # Update the last processed position
            last_pos = end

        # Add any remaining text after the last image
        if last_pos < len(text):
            contents.append({"type": "text", "text": text[last_pos:]})

        return contents

    def to_chat(
        self,
        system_prompt: str,
        instruction: str,
        source: str,
        target_prefix: str,
        demos: List[Dict[str, Any]],
        media: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        messages = []

        if system_prompt or instruction:
            system_content = self.to_content(
                system_prompt + ("\n" if system_prompt != "" else "") + instruction,
                media,
            )
            messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )

        for demo_instance in demos:
            user_content = self.to_content(demo_instance["source"], media)
            assistant_content = self.to_content(
                target_prefix + demo_instance["target"], media
            )
            messages.extend(
                [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]
            )

        last_user_content = self.to_content(source, media)

        messages.extend([{"role": "user", "content": last_user_content}])

        return messages

    def _format_instance_to_source(
        self,
        system_prompt: str,
        instruction: str,
        source: str,
        target_prefix: str,
        demos: List[Dict[str, Any]],
        media: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[Message]]:
        chat = self.to_chat(
            system_prompt,
            instruction,
            source,
            target_prefix,
            demos,
            media,
        )
        media["images"] = []
        return chat


class HFSystemFormat(ChatAPIFormat):
    r"""Formats the complete input for the model using the HuggingFace chat template of a given model.

    HFSystemFormat formats instance fields into a single string to be inputted to the model. This string overwrites
    field "source" of the instance.

    Example:
        ``HFSystemFormat(model_name="HuggingFaceH4/zephyr-7b-beta")``  Uses the template defined the in tokenizer_config.json of the model:

        ``"chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"``

        See more details in https://huggingface.co/docs/transformers/main/en/chat_templating

    """

    model_name: str
    _requirements_list = ["transformers", "Jinja2"]

    @retry_connection_with_exponential_backoff(backoff_factor=2)
    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _format_instance_to_source(
        self,
        system_prompt: str,
        instruction: str,
        source: str,
        target_prefix: str,
        demos: List[Dict[str, Any]],
        media: Optional[Dict[str, Any]] = None,
    ) -> str:
        chat = self.to_chat(
            system_prompt, instruction, source, target_prefix, demos, media
        )
        return (
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            + target_prefix
        )
