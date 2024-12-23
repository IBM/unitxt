import re

from ..operators import FieldOperator


class AddPrefix(FieldOperator):
    prefix: str

    def process_value(self, text: str) -> str:
        text = text.strip()
        if text.startswith(self.prefix):
            return text
        return self.prefix + text.strip()


class GetSQL(FieldOperator):
    def process_value(self, text: str) -> str:
        """Extracts the first SQL query from a given text.

        Args:
        text: The input string containing the SQL query.

        Returns:
        The first SQL query found in the text, or None if no query is found.
        """
        match = re.search(
            r"(?:```)?.*?(SELECT.*?(?:FROM|WITH|;|$).*?)(?:```|;|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if match:
            out = (
                text[match.start() : match.end()]
                .replace("```", "")
                .replace(";", "")
                .strip()
            )
        else:
            out = "No query found in generation"

        return out
