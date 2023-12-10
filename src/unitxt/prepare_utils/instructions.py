from ..instructions import TextualInstruction

sentence_classification_instructions = [
    "Predict the class of the following ({choices}):",
    "What is the type of the following? Types:{choices}",
    "Which of the choices {choices}, best describes the following text:",
]
# instructions?
sentence_classification_instructions = [
    TextualInstruction(text=x) for x in sentence_classification_instructions
]
