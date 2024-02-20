from .artifact import Artifact


class TextualInstruction(Artifact):
    """The role of TextualInstruction is to arrange potential instructions in the catalog, expressed as formatting strings.

    The (formatted) instructions are added to the instances, in field named "instruction" via the Template Operator.

    """

    text: str

    def get_instruction(self) -> str:
        return self.text
