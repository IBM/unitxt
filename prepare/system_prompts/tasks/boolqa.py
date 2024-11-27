from unitxt.catalog import add_to_catalog
from unitxt.system_prompts import TextualSystemPrompt

system_prompt = TextualSystemPrompt(
    "You are an agent in charge of answering a boolean (yes/no) question. The system presents "
    "you with a passage and a question. Read the passage carefully, and then answer yes or no. "
    "Think about your answer, and make sure it makes sense. Do not explain the answer. "
    "Only say yes or no.",
    __deprecated_msg__="This legacy system prompt reflects a task specific instruction, which is best handled by the 'instruction' field of the template.",
)

add_to_catalog(system_prompt, "system_prompts.boolqa", overwrite=True)
