from unitxt import add_to_catalog
from unitxt.standard import StandardRecipe

subsets = {  # the key must appear in the card name
    "cards.legalbench": [
        "abercrombie",
        "proa",
        "function_of_decision_section",
        "international_citizenship_questions",
        "corporate_lobbying",
    ],
    "cards.mmlu_pro": [
        "history",
        "law",
        "health",
        "physics",
        "business",
        "other",
        "philosophy",
        "psychology",
        "economics",
        "math",
        "biology",
        "chemistry",
        "computer_science",
        "engineering",
    ],
    "cards.safety.bbq": [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_SES",
        "Race_x_gender",
        "Religion",
        "SES",
        "Sexual_orientation",
    ],
    "cards.CFPB.product.watsonx": ["watsonx", "2023"],
    "cards.universal_ner": ["en.ewt"],  # , "en.pud"],
    "cards.mt.flores_101": [
        "ara_eng",
        "deu_eng",
        "eng_ara",
        "eng_deu",
        "eng_fra",
        "eng_kor",
        "eng_por",
        "eng_ron",
        "eng_spa",
        "fra_eng",
        "jpn_eng",
        "kor_eng",
        "por_eng",
        "ron_eng",
        "spa_eng",
    ],
}

default_args = {
    "demos_pool_size": 100,
    "num_demos": 1,
    "demos_taken_from": "train",
    "template_card_index": 1,
    "max_train_instances": 1000,
    "max_validation_instances": 1000,
    "max_test_instances": 1000,
}


def prepapre_recipe(default_args, specific_args):
    recipe = {}
    recipe.update(default_args)
    recipe.update(specific_args)

    for parent_card in subsets:  # Iterate directly through keys
        if parent_card in recipe["card"]:
            recipe["max_test_instances"] //= len(subsets[parent_card])
            break  # Exit the loop once a match is found

    if "template" in recipe and "template_card_index" in recipe:
        del recipe["template_card_index"]
    return StandardRecipe(**recipe)


### Reasoning

ingridients = {
    "card": "cards.hellaswag",
    "num_demos": 5,
    "template": "templates.completion.multiple_choice.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, "recipes.bluebench.reasoning.hellaswag", overwrite=True)

ingridients = {
    "card": "cards.openbook_qa",
    "num_demos": 5,
    "template": "templates.qa.multiple_choice.open.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, "recipes.bluebench.reasoning.openbook_qa", overwrite=True)


### Translation

for subset in subsets["cards.mt.flores_101"]:
    ingridients = {
        "card": f"cards.mt.flores_101.{subset}",
        "num_demos": 5,
        "template": "templates.translation.directed.bluebench",
        "demos_taken_from": "validation",
    }
    recipe = prepapre_recipe(default_args, ingridients)
    add_to_catalog(
        recipe,
        f'recipes.bluebench.translation.mt_flores_101_{subset.replace(".", "_").lower()}',
        overwrite=True,
    )


### Chatbot_abilities

ingridients = {
    "card": "cards.arena_hard.generation.english_gpt_4_0314_reference",
    "demos_pool_size": 0,
    "num_demos": 0,
    "template": "templates.empty",
    "metrics": [
        "metrics.llm_as_judge.pairwise_comparative_rating.llama_3_70b_instruct_generic_template_arena_hard"
    ],
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe,
    "recipes.bluebench.chatbot_abilities.arena_hard_generation_english_gpt_4_0314_reference",
    overwrite=True,
)


### News_classification

ingridients = {
    "card": "cards.20_newsgroups",
    "template": "templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe, "recipes.bluebench.news_classification.20_newsgroups", overwrite=True
)


### Bias

for subset in subsets["cards.safety.bbq"]:
    ingridients = {
        "card": f"cards.safety.bbq.{subset}",
        "demos_pool_size": 20,
        "num_demos": 5,
        "template": "templates.qa.multiple_choice.with_context.match",
        "demos_taken_from": "test",
    }
    recipe = prepapre_recipe(default_args, ingridients)
    add_to_catalog(
        recipe,
        f'recipes.bluebench.bias.safety_bbq_{subset.replace(".", "_").lower()}',
        overwrite=True,
    )


### Legal

for subset in subsets["cards.legalbench"]:
    ingridients = {
        "card": f"cards.legalbench.{subset}",
        "demos_pool_size": 10,
        "template": "templates.classification.multi_class.bluebench",
        "demos_taken_from": "test",
    }
    recipe = prepapre_recipe(default_args, ingridients)
    add_to_catalog(
        recipe,
        f'recipes.bluebench.legal.legalbench_{subset.replace(".", "_").lower()}',
        overwrite=True,
    )


### Product_help

ingridients = {
    "card": "cards.CFPB.product.watsonx",
    "num_demos": 5,
    "template": "templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe, "recipes.bluebench.product_help.cfpb_product_watsonx", overwrite=True
)

ingridients = {
    "card": "cards.CFPB.product.2023",
    "num_demos": 5,
    "template": "templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe, "recipes.bluebench.product_help.cfpb_product_2023", overwrite=True
)


### Knowledge

for subset in subsets["cards.mmlu_pro"]:
    ingridients = {
        "card": f"cards.mmlu_pro.{subset}",
        "demos_pool_size": 20,
        "num_demos": 5,
        "template": "templates.qa.multiple_choice.with_topic.bluebench",
        "demos_taken_from": "test",
    }
    recipe = prepapre_recipe(default_args, ingridients)
    add_to_catalog(
        recipe,
        f'recipes.bluebench.knowledge.mmlu_pro_{subset.replace(".", "_").lower()}',
        overwrite=True,
    )


### Entity_extraction

for subset in subsets["cards.universal_ner"]:
    ingridients = {
        "card": f"cards.universal_ner.{subset}",
        "demos_pool_size": 10000,
        "num_demos": 5,
        "template": "templates.span_labeling.extraction.title",
        "metrics": ["metrics.ner[zero_division=1.0]"],
        "train_refiner": "operators.balancers.ner.zero_vs_many_entities[segments_boundaries=[0,1,2]]",
        "demos_taken_from": "test" if "pud" in subset else "train",
    }
    recipe = prepapre_recipe(default_args, ingridients)
    add_to_catalog(
        recipe,
        f'recipes.bluebench.entity_extraction.universal_ner_{subset.replace(".", "_").lower()}',
        overwrite=True,
    )


### Safety

ingridients = {
    "card": "cards.attaq_500",
    "demos_pool_size": 0,
    "num_demos": 0,
    "template_card_index": 0,
    "max_test_instances": 100,
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, "recipes.bluebench.safety.attaq_500", overwrite=True)


### Summarization

ingridients = {
    "card": "cards.billsum_document_filtered_to_6000_chars",
    "num_demos": 0,
    "template": "templates.summarization.abstractive.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe,
    "recipes.bluebench.summarization.billsum_document_filtered_to_6000_chars",
    overwrite=True,
)

ingridients = {
    "card": "cards.tldr_document_filtered_to_6000_chars",
    "num_demos": 0,
    "template": "templates.summarization.abstractive.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe,
    "recipes.bluebench.summarization.tldr_document_filtered_to_6000_chars",
    overwrite=True,
)


### RAG_general

ingridients = {
    "card": "cards.rag.response_generation.clapnq",
    "template": "templates.rag.response_generation.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(
    recipe,
    "recipes.bluebench.rag_general.rag_response_generation_clapnq",
    overwrite=True,
)


### QA_finance

ingridients = {
    "card": "cards.fin_qa",
    "num_demos": 1,
    "template_card_index": 0,
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, "recipes.bluebench.qa_finance.fin_qa", overwrite=True)
