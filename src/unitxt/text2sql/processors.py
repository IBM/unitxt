from ..operators import FieldOperator


def process_value(self, text: str) -> str:
    text = text.strip()
    text = text.removeprefix("```")
    text = text.removeprefix("sql")
    text = text.removesuffix("```")
    if "```" in text:
        text = text.split("```")[0]
    return text


class AddPrefix(FieldOperator):
    prefix: str

    def process_value(self, text: str) -> str:
        text = text.strip()
        if text.startswith(self.prefix):
            return text
        return self.prefix + text.strip()


class GetSQL(FieldOperator):
    def process_value(self, text: str) -> str:
        text = text.strip()
        if "\n\n" in text:
            text = text.split("\n\n")[0]
        if "<|eot_id|>" in text:
            text = text[: text.find("<|eot_id|>")]
        if "SELECT" in text and ";" in text:
            return text[text.find("SELECT") : text.find(";") + 1]
        if "SELECT" in text:
            return text[text.find("SELECT") :]
        return text
