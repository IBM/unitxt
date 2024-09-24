from unitxt import add_to_catalog
from unitxt.standard import StandardRecipe
from datasets import load_dataset

subsets = {  # the key must appear in the card name
    "legalbench": [
        "abercrombie",
        "proa",
        "function_of_decision_section",
        "international_citizenship_questions",
        "corporate_lobbying",
    ],
    "mmlu_pro": [
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
    "bbq": [
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
    "CFPB.product": ["watsonx", "2023"],
    "universal_ner": ["en.ewt"],  # , "en.pud"],
    "flores_101": [
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
    "max_test_instances": 100
}


def prepapre_recipe(default_args, specific_args):
    recipe = {}
    recipe.update(default_args)
    recipe.update(specific_args)
    if "template" in recipe and "template_card_index" in recipe:
        del recipe["template_card_index"]
    return StandardRecipe(**recipe)

### Reasoning

ingridients = {
               "card": f"cards.hellaswag",
               "num_demos": 5,
               "template": f"templates.completion.multiple_choice.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.reasoning.hellaswag", overwrite=True)

ingridients = {
               "card": f"cards.openbook_qa",
               "num_demos": 5,
               "template": f"templates.qa.multiple_choice.open.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.reasoning.openbook_qa", overwrite=True)


### Translation

for subset in subsets["flores_101"]:
        ingridients = {
                       "card": f"cards.mt.flores_101.{subset}",
                       "num_demos": 5,
                       "template": f"templates.translation.directed.bluebench",
                       "demos_taken_from": f"validation",
        }
        recipe = prepapre_recipe(default_args, ingridients)
        add_to_catalog(recipe, f"recipes.bluebench.translation.mt_flores_101_{subset.replace(".", "_").lower()}", overwrite=True)


### Chatbot_abilities

ingridients = {
               "card": f"cards.arena_hard.generation.english_gpt_4_0314_reference",
               "demos_pool_size": 0,
               "num_demos": 0,
               "template": f"templates.empty",
               "metrics": ['metrics.llm_as_judge.pairwise_comparative_rating.llama_3_70b_instruct_ibm_genai_template_arena_hard'],
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.chatbot_abilities.arena_hard_generation_english_gpt_4_0314_reference", overwrite=True)


### News_classification

ingridients = {
               "card": f"cards.20_newsgroups",
               "template": f"templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.news_classification.20_newsgroups", overwrite=True)


### Bias

for subset in subsets["bbq"]:
        ingridients = {
                       "card": f"cards.safety.bbq.{subset}",
                       "demos_pool_size": 20,
                       "num_demos": 5,
                       "template": f"templates.qa.multiple_choice.with_context.match",
                       "demos_taken_from": f"test",
        }
        recipe = prepapre_recipe(default_args, ingridients)
        add_to_catalog(recipe, f"recipes.bluebench.bias.safety_bbq_{subset.replace(".", "_").lower()}", overwrite=True)


### Legal

for subset in subsets["legalbench"]:
        ingridients = {
                       "card": f"cards.legalbench.{subset}",
                       "demos_pool_size": 10,
                       "template": f"templates.classification.multi_class.bluebench",
                       "demos_taken_from": f"test",
        }
        recipe = prepapre_recipe(default_args, ingridients)
        add_to_catalog(recipe, f"recipes.bluebench.legal.legalbench_{subset.replace(".", "_").lower()}", overwrite=True)


### Product_help

ingridients = {
               "card": f"cards.CFPB.product.watsonx",
               "num_demos": 5,
               "template": f"templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.product_help.cfpb_product_watsonx", overwrite=True)

ingridients = {
               "card": f"cards.CFPB.product.2023",
               "num_demos": 5,
               "template": f"templates.classification.multi_class.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.product_help.cfpb_product_2023", overwrite=True)


### Knowledge

for subset in subsets["mmlu_pro"]:
        ingridients = {
                       "card": f"cards.mmlu_pro.{subset}",
                       "demos_pool_size": 20,
                       "num_demos": 5,
                       "template": f"templates.qa.multiple_choice.with_topic.bluebench",
                       "demos_taken_from": f"test",
        }
        recipe = prepapre_recipe(default_args, ingridients)
        add_to_catalog(recipe, f"recipes.bluebench.knowledge.mmlu_pro_{subset.replace(".", "_").lower()}", overwrite=True)


### Entity_extraction

for subset in subsets["universal_ner"]:
        ingridients = {
                       "card": f"cards.universal_ner.{subset}",
                       "demos_pool_size": 10000,
                       "num_demos": 5,
                       "template": f"templates.span_labeling.extraction.title",
        }
        recipe = prepapre_recipe(default_args, ingridients)
        add_to_catalog(recipe, f"recipes.bluebench.entity_extraction.universal_ner_{subset.replace(".", "_").lower()}", overwrite=True)


### Safety

ingridients = {
               "card": f"cards.attaq_500",
               "demos_pool_size": 0,
               "num_demos": 0,
               "template_card_index": 0,
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.safety.attaq_500", overwrite=True)


### Summarization

ingridients = {
               "card": f"cards.billsum_document_filtered_to_6000_chars",
               "num_demos": 0,
               "template": f"templates.summarization.abstractive.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.summarization.billsum_document_filtered_to_6000_chars", overwrite=True)

ingridients = {
               "card": f"cards.tldr_document_filtered_to_6000_chars",
               "num_demos": 0,
               "template": f"templates.summarization.abstractive.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.summarization.tldr_document_filtered_to_6000_chars", overwrite=True)


### RAG_general

ingridients = {
               "card": f"cards.rag.response_generation.clapnq",
               "template": f"templates.rag.response_generation.bluebench",
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.rag_general.rag_response_generation_clapnq", overwrite=True)


### QA_finance

ingridients = {
               "card": f"cards.fin_qa",
               "num_demos": 2,
               "template_card_index": 0,
}
recipe = prepapre_recipe(default_args, ingridients)
add_to_catalog(recipe, f"recipes.bluebench.qa_finance.fin_qa", overwrite=True)