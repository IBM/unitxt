from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        instruction="""Read the following two parts: (1) The Conversation between the user and the agent, which can include multiple turns. The two speakers (user and agent) alternate in the conversation, and the user poses an inquiry at the end. (2) The Response (of the agent) to the last turn user query that continues the conversation. The response can rely on all the information provided in the previous turns. Your task is to evaluate if the Response (part 2) only contain relevant information to the inquiry, using one of three labels: [yes, no, unsure] """,
        input_format="\n\nConversation:\n{question}\n\nResponse:\n{answer}\n\n\nOutput:",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_string_judgment",
        ],
    ),
    "templates.response_assessment.judges.relevance.v3",
    overwrite=True,
)
