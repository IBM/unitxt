import unitxt
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadJsonFile
from unitxt.operators import (
    Copy,
    ExecuteExpression,
)
from unitxt.stream_operators import JoinStreams
from unitxt.test_utils.card import test_card

# ACEBench datasets
ace_base_path = (
    "https://raw.githubusercontent.com/ACEBench/ACEBench/main/data_all/data_en/"
)
ace_answer_path = "https://raw.githubusercontent.com/ACEBench/ACEBench/main/data_all/data_en/possible_answer/"

with unitxt.settings.context(allow_unverified_code=True):
    # Single turn subsets with ground truth data
    for subset in [
        "normal_single_turn_single_function",
        "normal_single_turn_parallel_function",
        "normal_atom_bool",
        "normal_atom_enum",
        "normal_atom_list",
        "normal_atom_number",
        "normal_atom_object_deep",
        "normal_atom_object_short",
        "normal_preference",
        "normal_similar_api",
    ]:
        card = TaskCard(
            loader=LoadJsonFile(
                files={
                    "questions": ace_base_path + f"data_{subset}.json",
                    "answers": ace_answer_path + f"data_{subset}.json",
                },
                lines=True,
                data_classification_policy=["public"],
            ),
            preprocess_steps=[
                JoinStreams(
                    left_stream="questions",
                    right_stream="answers",
                    how="inner",
                    on="id",
                    new_stream_name="test",
                ),
                Copy(field="question", to_field="query"),
                Copy(field="function", to_field="tools"),
                "operators.normalize_tool_schema",
                "operators.fix_json_schema",
                ExecuteExpression(
                    expression='[{"name": k, "arguments": v} for k, v in ground_truth.items()]',
                    to_field="reference_calls",
                ),
            ],
            task="tasks.tool_calling.supervised",
            templates=["templates.tool_calling.base"],
            __description__=(
                """ACEBench is a comprehensive benchmark designed to evaluate Large Language Models' capabilities in tool use across various scenarios. It provides a structured approach to assess models' performance in understanding and utilizing APIs across 68 sub-domains within 8 major domains, including technology, finance, entertainment, society, health, culture, and environment."""
            ),
            __title__=f"""ACEBench - {subset.replace("_", " ").title()}""",
            __tags__={
                "annotations_creators": "expert-generated",
                "language": ["en"],
                "license": "apache-2.0",
                "size_categories": ["1K<n<10K"],
                "task_categories": [
                    "question-answering",
                    "tool-calling",
                ],
                "task_ids": ["tool-calling"],
            },
        )

        # Test and add the card to the catalog
        test_card(card, strict=False)
        add_to_catalog(card, f"cards.acebench.{subset}", overwrite=True)

    # Multi-turn subsets with ground truth data
    for subset in [
        "normal_multi_turn_user_adjust",
        "normal_multi_turn_user_switch",
        "agent_multi_step",
        "agent_multi_turn",
    ]:
        card = TaskCard(
            loader=LoadJsonFile(
                files={
                    "questions": ace_base_path + f"data_{subset}.json",
                    "answers": ace_answer_path + f"data_{subset}.json",
                },
                lines=True,
                data_classification_policy=["public"],
            ),
            preprocess_steps=[
                JoinStreams(
                    left_stream="questions",
                    right_stream="answers",
                    how="inner",
                    on="id",
                    new_stream_name="test",
                ),
                ExecuteExpression(
                    expression='[{"role": turn.split(":", 1)[0].strip(), "content": turn.split(":", 1)[1].strip()} for turn in question.split("\\n") if ":" in turn and turn.split(":", 1)[0].strip() in ["user", "system"]]',
                    to_field="dialog",
                ),
                Copy(field="function", to_field="tools"),
                "operators.normalize_tool_schema",
                "operators.fix_json_schema",
                ExecuteExpression(
                    expression='[{"name": k, "arguments": v} for k, v in ground_truth.items()] if isinstance(ground_truth, dict) else []',
                    to_field="reference_calls",
                ),
            ],
            task="tasks.tool_calling.multi_turn",
            templates=["templates.tool_calling.multi_turn"],
            __description__=(
                """ACEBench is a comprehensive benchmark designed to evaluate Large Language Models' capabilities in tool use across various scenarios. This multi-turn variant tests models' ability to handle complex, interactive scenarios requiring multiple exchanges."""
            ),
            __title__=f"""ACEBench Multi-Turn - {subset.replace("_", " ").title()}""",
            __tags__={
                "annotations_creators": "expert-generated",
                "language": ["en"],
                "license": "apache-2.0",
                "size_categories": ["1K<n<10K"],
                "task_categories": [
                    "question-answering",
                    "tool-calling",
                    "multi-turn-tool-calling",
                ],
                "task_ids": [
                    "tool-calling",
                    "multi-turn-tool-calling",
                ],
            },
        )

        # Test and add the card to the catalog
        test_card(card, strict=False)
        add_to_catalog(card, f"cards.acebench.multi_turn.{subset}", overwrite=True)

    # Special subsets with ground truth data (these may have different ground truth formats)
    for subset in [
        "special_error_param",
        "special_incomplete",
        "special_irrelevant",
    ]:
        card = TaskCard(
            loader=LoadJsonFile(
                files={
                    "questions": ace_base_path + f"data_{subset}.json",
                    "answers": ace_answer_path + f"data_{subset}.json",
                },
                lines=True,
                data_classification_policy=["public"],
            ),
            preprocess_steps=[
                JoinStreams(
                    left_stream="questions",
                    right_stream="answers",
                    how="inner",
                    on="id",
                    new_stream_name="test",
                ),
                Copy(field="question", to_field="query"),
                Copy(field="function", to_field="tools"),
                "operators.normalize_tool_schema",
                "operators.fix_json_schema",
                ExecuteExpression(
                    expression='[{"name": k, "arguments": v} for k, v in ground_truth.items()] if isinstance(ground_truth, dict) and ground_truth else []',
                    to_field="reference_calls",
                ),
            ],
            task="tasks.tool_calling.supervised",
            templates=["templates.tool_calling.base"],
            __description__=(
                """ACEBench is a comprehensive benchmark designed to evaluate Large Language Models' capabilities in tool use across various scenarios. This special variant tests models' handling of challenging cases including error parameters, incomplete information, and irrelevant requests."""
            ),
            __title__=f"""ACEBench Special - {subset.replace("_", " ").title()}""",
            __tags__={
                "annotations_creators": "expert-generated",
                "language": ["en"],
                "license": "apache-2.0",
                "size_categories": ["1K<n<10K"],
                "task_categories": [
                    "question-answering",
                    "tool-calling",
                ],
                "task_ids": ["tool-calling"],
            },
        )

        # Test and add the card to the catalog
        test_card(card, strict=False)
        add_to_catalog(card, f"cards.acebench.special.{subset}", overwrite=True)
