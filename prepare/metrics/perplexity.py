from src.unitxt import add_to_catalog
from src.unitxt.metrics import Perplexity
from src.unitxt.test_utils.metrics import test_metric


def run_test(metric_to_test, instance_scores, global_scores):
    references = []
    predictions = []
    instance_targets = []

    for r, p, pr in instance_scores:
        references.append([r])
        predictions.append(p)
        instance_targets.append(
            {
                "perplexity": pr,
                "score": pr,
                "score_name": "perplexity",
                "reference_scores": [pr],
            }
        )

    global_target = {
        "perplexity": global_scores["mean"],
        "perplexity_ci_high": global_scores["ci_high"],
        "perplexity_ci_low": global_scores["ci_low"],
        "score": global_scores["mean"],
        "score_ci_high": global_scores["ci_high"],
        "score_ci_low": global_scores["ci_low"],
        "score_name": "perplexity",
    }

    test_metric(
        metric=metric_to_test,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
    )


def flatten(instances: dict, flip: bool):
    result = []
    for q, answers in instances.items():
        for a, pr in answers:
            if flip:
                result.append((a, q, pr))
            else:
                result.append((q, a, pr))
    return result


perplexity = Perplexity(
    model_name="google/flan-t5-small", perplexity_prompt="Complete the given content:"
)

perplexity_question = Perplexity(
    model_name="google/flan-t5-small",
    perplexity_prompt="Generate a question based on the given content:",
)

perplexity_answer = Perplexity(
    model_name="google/flan-t5-small",
    perplexity_prompt="Generate an answer based on the given content:",
)

perplexity_chat = Perplexity(
    model_name="google/flan-t5-small",
    perplexity_prompt="Generate a conversation between a user and an agent "
    "based on the given content:",
)

perplexity_chat_bloom = Perplexity(
    model_name="bigscience/bloom-560M",
    perplexity_prompt="Generate a conversation between a user and an agent "
    "based on the given content:",
)

gen_instances = flatten(
    {
        "together we are": [  # prediction 1
            ("ok", 0.04),  # reference 1
            ("stronger", 0.01),  # reference 2
            ("weaker", 0.01),
        ],
        "The future belongs to ": [
            ("those who prepare for it today", 0.02),
            ("our youth", 0.01),
            ("those who believe in the beauty of their dreams", 0.06),
        ],
    },
    flip=True,
)

gen_global = {
    "mean": 0.03,
    "ci_high": 0.05,
    "ci_low": 0.01,
}

q_instances = flatten(
    {
        "who are we?": [  # reference 1
            ("we are the world", 0.06),  # prediction 1
            ("we are the children", 0.07),  # prediction 2
            ("we are the people", 0.08),  # prediction 3
        ],
        "what are we saving?": [
            ("we make a brighter day", 0.07),
            ("we are making a choice", 0.06),
            ("we are saving our own lives", 0.26),
        ],
        "Which city is the capital of Italy?": [
            ("Milan", 0.21),
            ("Rome", 0.22),
            ("Naples", 0.2),
        ],
    },
    flip=False,
)
q_global = {
    "mean": 0.14,
    "ci_high": 0.19,
    "ci_low": 0.08,
}

a_instances = flatten(
    {
        "Which city is the capital of Italy?": [
            ("The capital of Italy is Milan", 0.08),
            ("The capital of Italy is Rome", 0.08),
            ("The capital of Italy is Naples", 0.05),
        ],
        "Where Albert Einstein was born?": [
            ("Einstein was born in Ulm, Germany", 0.07),
            ("Einstein was born in Ulm, Poland", 0.04),
            ("Ulm", 0.0),
            ("Germany", 0.12),
        ],
    },
    flip=True,
)
a_global = {
    "mean": 0.06,
    "ci_high": 0.09,
    "ci_low": 0.04,
}

chat_pension_policy = (
    "Pension policy refers to the mix of public and private programs that provide income to an individual or "
    "his/her survivors during retirement or incapacity. Pensions take myriad forms: they may be flat-rate, "
    "earnings related, lump sum, or annuities. In most affluent democracies, state provision dominates pension "
    "policy, usually supplemented by private pensions that are part of employment contracts or individual "
    "private pension products. This mix of public and private provision is central to understanding the "
    "historical development of pension policy and national responses to the current dilemmas presented by "
    "aging and fiscal austerity."
)

chat_retirement_policy = (
    "The employee retirement policy outlines the guidelines and procedures for employee retirement from your "
    "organization It is designed to provide a smooth transition for employees who are approaching retirement "
    "age and to ensure that their departure does not disrupt the normal operations of the organization."
)

chat_construction_policy = (
    "According to new land use and construction policies, structural tsunami mitigation measures are no longer "
    "considered fail-proof solutions, regardless of their protection levels. Rather, communities operate under "
    "the assumption that engineered prevention structures will be effective only in rare yet more likely "
    "tsunami events, understanding that unforeseen event magnitudes are always possible, given the Great East "
    "Japan Earthquake experience."
)

chat_instances = flatten(
    {
        "user: hello\nagent:I have a question about my retirement policy.": [
            (chat_pension_policy, 0.02),
            (chat_retirement_policy, 0.03),
            (chat_construction_policy, 0.01),
        ],
    },
    flip=False,
)
chat_global = {
    "mean": 0.02,
    "ci_high": 0.03,
    "ci_low": 0.01,
}

chat_instances_bloom = flatten(
    {
        "user: hello\nagent:I have a question about my retirement policy.": [
            (chat_pension_policy, 0.01),
            (chat_retirement_policy, 0.02),
            (chat_construction_policy, 0.01),
        ],
    },
    flip=False,
)
chat_global_bloom = {
    "mean": 0.01,
    "ci_high": 0.02,
    "ci_low": 0.01,
}

run_test(perplexity, gen_instances, gen_global)
run_test(perplexity_question, q_instances, q_global)
run_test(perplexity_answer, a_instances, a_global)
run_test(perplexity_chat, chat_instances, chat_global)
run_test(perplexity_chat_bloom, chat_instances_bloom, chat_global_bloom)

add_to_catalog(perplexity, "metrics.perplexity.flan_t5_small", overwrite=True)
add_to_catalog(
    perplexity_question, "metrics.perplexity_q.flan_t5_small", overwrite=True
)
add_to_catalog(perplexity_answer, "metrics.perplexity_a.flan_t5_small", overwrite=True)
add_to_catalog(perplexity_chat, "metrics.perplexity_chat.flan_t5_small", overwrite=True)
