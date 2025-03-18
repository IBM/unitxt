from unitxt import add_to_catalog
from unitxt.metrics import Perplexity
from unitxt.test_utils.metrics import test_metric

skip_nli_metric_test = True


def run_test(metric_to_test, instance_scores, global_scores):
    references = []
    predictions = []
    instance_targets = []

    for instance in instance_scores:
        score = instance["score"]
        references.append(instance["references"])
        predictions.append(instance["prediction"])
        instance_targets.append(
            {
                "perplexity": score,
                "score": score,
                "score_name": "perplexity",
                "reference_scores": [score],
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
        "num_of_instances": len(instance_scores),
    }

    test_metric(
        metric=metric_to_test,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
    )


def generate_completions(instances, global_scores):
    instance_scores = []
    for opening, completions in instances.items():
        for completion, pr in completions:
            instance_scores.append(
                {"prediction": completion, "references": [opening], "score": pr}
            )
    run_test(perplexity, instance_scores, global_scores)


def generate_answers(instances, global_scores):
    instance_scores = []
    for question, answers in instances.items():
        for answer, pr in answers:
            instance_scores.append(
                {"prediction": answer, "references": [question], "score": pr}
            )
    run_test(perplexity_answer, instance_scores, global_scores)


def generate_questions(instances, global_scores, metric):
    instance_scores = []
    for question, answers in instances.items():
        for answer, pr in answers:
            instance_scores.append(
                {"prediction": question, "references": [answer], "score": pr}
            )
    run_test(metric, instance_scores, global_scores)


def generate_nli(instances, global_scores, metric):
    if skip_nli_metric_test:
        return
    instance_scores = []
    for premise, hypotheses in instances.items():
        for hypothesis, pr in hypotheses:
            instance_scores.append(
                {"prediction": hypothesis, "references": [premise], "score": pr}
            )
    run_test(metric, instance_scores, global_scores)


perplexity = Perplexity(
    model_name="google/flan-t5-small",
    source_template="Complete the given content: {reference}",
    target_template="{prediction}",
)

perplexity_question = Perplexity(
    model_name="google/flan-t5-small",
    source_template="Generate a question based on the given content: {reference}",
    target_template="{prediction}",
)

perplexity_answer = Perplexity(
    model_name="google/flan-t5-small",
    source_template="Generate an answer based on the given content: {reference}",
    target_template="{prediction}",
)

perplexity_chat = Perplexity(
    model_name="google/flan-t5-small",
    source_template="Generate a conversation between a user and an agent "
    "based on the given content: {reference}",
    target_template="{prediction}",
)

perplexity_chat_bloom = Perplexity(
    model_name="bigscience/bloom-560M",
    source_template="Generate a conversation between a user and an agent "
    "based on the given content: {reference}",
    target_template="{prediction}",
)

perplexity_nli = Perplexity(
    model_name="google/t5_xxl_true_nli_mixture",
    source_template="premise: {reference} hypothesis: {prediction}",
    target_template="1",
    single_token_mode=True,
)


generate_completions(
    instances={
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
    global_scores={
        "mean": 0.03,
        "ci_high": 0.05,
        "ci_low": 0.01,
        "num_of_instances": 6,
    },
)

generate_questions(
    instances={
        "who are we?": [  # prediction
            ("we are the world", 0.06),  # reference 1
            ("we are the children", 0.07),  # reference 2
            ("we are the people", 0.08),  # reference 3
        ],
        "what are we saving?": [
            ("we make a brighter day", 0.07),
            ("we are making a choice", 0.06),
            ("we are saving our own lives", 0.26),
        ],
        "Which city is the capital of Italy?": [
            ("The capital of Italy is Milan", 0.54),
            ("The capital of England is London", 0.2),
            ("The capital of Spain is Madrid", 0.22),
        ],
    },
    global_scores={
        "mean": 0.17,
        "ci_high": 0.31,
        "ci_low": 0.1,
    },
    metric=perplexity_question,
)

generate_answers(
    instances={
        "Where was Einstein born?": [
            ("Einstein was born in Ulm, Germany", 0.07),
            ("Einstein was raised in Ulm, Poland", 0.02),
            ("Ulm", 0.0),
            ("Germany", 0.15),
        ],
    },
    global_scores={
        "mean": 0.06,
        "ci_high": 0.13,
        "ci_low": 0.01,
    },
)

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

generate_questions(
    instances={
        "user: hello\nagent:I have a question about my retirement policy.": [
            (chat_pension_policy, 0.02),
            (chat_retirement_policy, 0.03),
            (chat_construction_policy, 0.01),
        ],
    },
    global_scores={
        "mean": 0.02,
        "ci_high": 0.03,
        "ci_low": 0.01,
    },
    metric=perplexity_chat,
)

generate_questions(
    instances={
        "user: hello\nagent:I have a question about my retirement policy.": [
            (chat_pension_policy, 0.01),
            (chat_retirement_policy, 0.02),
            (chat_construction_policy, 0.01),
        ],
    },
    global_scores={
        "mean": 0.01,
        "ci_high": 0.02,
        "ci_low": 0.01,
    },
    metric=perplexity_chat_bloom,
)

generate_nli(
    instances={
        "Einstein is from Germany": [
            ("Einstein is from Europe", 0.72),
            ("Einstein is Jewish", 0.0),
            ("Einstein is from Alaska", 0.0),
        ],
        "Roosevelt is from France": [("Roosevelt is from Paris", 0.06)],
        "Roosevelt is from Paris": [("Roosevelt is from France", 0.67)],
    },
    global_scores={
        "mean": 0.29,
        "ci_high": 0.57,
        "ci_low": 0.01,
    },
    metric=perplexity_nli,
)

add_to_catalog(perplexity, "metrics.perplexity.flan_t5_small", overwrite=True)
add_to_catalog(
    perplexity_question, "metrics.perplexity_q.flan_t5_small", overwrite=True
)
add_to_catalog(perplexity_answer, "metrics.perplexity_a.flan_t5_small", overwrite=True)
add_to_catalog(perplexity_chat, "metrics.perplexity_chat.flan_t5_small", overwrite=True)
add_to_catalog(perplexity_nli, "metrics.perplexity_nli.t5_nli_mixture", overwrite=True)
