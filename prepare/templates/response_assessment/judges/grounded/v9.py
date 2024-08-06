from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        instruction="""Read the following three parts: (1) Document, (2) the Inquiry which shows the conversation between the user and the agent and also one user query at the end, (3) the Response of the agent to the last user query of the conversation. Determine if the facts in the response (part 3) come entirely from the document text (part 1) without any external information. Return only one of the following answers: [yes, no, unsure].""",
        input_format="\n\nConversation:\n{question}\n\nResponse:\n{answer}\n\n\nOutput:",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_string_judgment",
        ],
    ),
    "templates.response_assessment.judges.grounded.v9",
    overwrite=True,
)
