import json
import time

import pandas as pd
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
    WMLInferenceEngineChat,
    WMLInferenceEngineGeneration,
)

print("Creating cross_provider_rits ...")
cross_provider_rits = CrossProviderInferenceEngine(
    model="granite-3-8b-instruct", max_tokens=32, provider="rits", temperature=0
)

print("Creating cross_provider_watsonx ...")
cross_provider_watsonx = CrossProviderInferenceEngine(
    model="granite-3-8b-instruct", max_tokens=32, provider="watsonx", temperature=0
)
print("Creating wml_gen ...")
wml_gen = WMLInferenceEngineGeneration(
    model_name="ibm/granite-3-8b-instruct", max_new_tokens=32, temperature=0
)
print("Creating wml_chat ...")
wml_chat = WMLInferenceEngineChat(
    model_name="ibm/granite-3-8b-instruct", max_tokens=32, temperature=0
)

df = pd.DataFrame(
    columns=[
        "model",
        "format",
        "system_prompt",
        "f1_micro",
        "ci_low",
        "ci_high",
        "duration",
        "num_instances",
        "type_of_input",
    ]
)

model_list = [
    (cross_provider_watsonx, "cross-provider-watsonx"),
    (wml_chat, "wml-chat"),
    (wml_gen, "wml-gen"),
]

# This example compares the impact of different formats on a classification dataset
#
# formats.chat_api  - creates a list of OpenAI messages, where the instruction appears in the system prompt.
#
# [
#    {
#        "role": "system",
#        "content": "Classify the contractual clauses of the following text to one of these options: Adjustments, Agreements, Amendments, Anti-Corruption Laws, Applicable Laws, Approvals, Arbitration, Assignments, Assigns, Authority, Authorizations, Base Salary, Benefits, Binding Effects, Books, Brokers, Capitalization, Change In Control, Closings, Compliance With Laws, Confidentiality, Consent To Jurisdiction, Consents, Construction, Cooperation, Costs, Counterparts, Death, Defined Terms, Definitions, Disability, Disclosures, Duties, Effective Dates, Effectiveness, Employment, Enforceability, Enforcements, Entire Agreements, Erisa, Existence, Expenses, Fees, Financial Statements, Forfeitures, Further Assurances, General, Governing Laws, Headings, Indemnifications, Indemnity, Insurances, Integration, Intellectual Property, Interests, Interpretations, Jurisdictions, Liens, Litigations, Miscellaneous, Modifications, No Conflicts, No Defaults, No Waivers, Non-Disparagement, Notices, Organizations, Participations, Payments, Positions, Powers, Publicity, Qualifications, Records, Releases, Remedies, Representations, Sales, Sanctions, Severability, Solvency, Specific Performance, Submission To Jurisdiction, Subsidiaries, Successors, Survival, Tax Withholdings, Taxes, Terminations, Terms, Titles, Transactions With Affiliates, Use Of Proceeds, Vacations, Venues, Vesting, Waiver Of Jury Trials, Waivers, Warranties, Withholdings."
#    },
#    {
#        "role": "user",
#        "content": "text: Each Credit Party shall maintain, in all material respects, proper, complete and consistent books of record with respect to such Person\u2019s operations, affairs, and financial condition.\u00a0From time to time upon reasonable prior notice, each Credit Party shall permit any Lender, at such reasonable times and intervals and to a reasonable extent and under the reasonable guidance of officers of or employees delegated by officers of such Credit Party, to, subject to any applicable confidentiality considerations, examine and copy the books and records of such Credit Party, to visit and inspect the Property of such Credit Party, and to discuss the business operations and Property of such Credit Party with the officers and directors thereof (provided that, so long as no Event of Default has occurred and is continuing, the Lenders shall be entitled to only one such visit per year coordinated by the Administrative Agent)."
#    },
#    {
#        "role": "assistant",
#        "content": "The contractual clauses is Records"
#    },
#    {
#        "role": "user",
#        "content": "text: Executive agrees to be employed with the Company, and the Company agrees to employ Executive, during the Term and on the terms and conditions set forth in this Agreement. Executive agrees during the term of this Agreement to devote substantially all of Executive\u2019s business time, efforts, skills and abilities to the performance of Executive\u2019s duties to the Company and to the furtherance of the Company's business."
#    }
# ]
#
# formats.chat_api[place_instruction_in_user_turns=True] - creates a list of OpenAI messages, where the instruction appears in each user turn prompt.
#
# [
#     {
#         "role": "user",
#         "content": "Classify the contractual clauses of the following text to one of these options: Adjustments, Agreements, Amendments, Anti-Corruption Laws, Applicable Laws, Approvals, Arbitration, Assignments, Assigns, Authority, Authorizations, Base Salary, Benefits, Binding Effects, Books, Brokers, Capitalization, Change In Control, Closings, Compliance With Laws, Confidentiality, Consent To Jurisdiction, Consents, Construction, Cooperation, Costs, Counterparts, Death, Defined Terms, Definitions, Disability, Disclosures, Duties, Effective Dates, Effectiveness, Employment, Enforceability, Enforcements, Entire Agreements, Erisa, Existence, Expenses, Fees, Financial Statements, Forfeitures, Further Assurances, General, Governing Laws, Headings, Indemnifications, Indemnity, Insurances, Integration, Intellectual Property, Interests, Interpretations, Jurisdictions, Liens, Litigations, Miscellaneous, Modifications, No Conflicts, No Defaults, No Waivers, Non-Disparagement, Notices, Organizations, Participations, Payments, Positions, Powers, Publicity, Qualifications, Records, Releases, Remedies, Representations, Sales, Sanctions, Severability, Solvency, Specific Performance, Submission To Jurisdiction, Subsidiaries, Successors, Survival, Tax Withholdings, Taxes, Terminations, Terms, Titles, Transactions With Affiliates, Use Of Proceeds, Vacations, Venues, Vesting, Waiver Of Jury Trials, Waivers, Warranties, Withholdings.
#                      text: Each Credit Party shall maintain, in all material respects, proper, complete and consistent books of record with respect to such Person\u2019s operations, affairs, and financial condition.\u00a0From time to time upon reasonable prior notice, each Credit Party shall permit any Lender, at such reasonable times and intervals and to a reasonable extent and under the reasonable guidance of officers of or employees delegated by officers of such Credit Party, to, subject to any applicable confidentiality considerations, examine and copy the books and records of such Credit Party, to visit and inspect the Property of such Credit Party, and to discuss the business operations and Property of such Credit Party with the officers and directors thereof (provided that, so long as no Event of Default has occurred and is continuing, the Lenders shall be entitled to only one such visit per year coordinated by the Administrative Agent)."
#     },
#     {
#         "role": "assistant",
#         "content": "The contractual clauses is Records"
#     },
#     {
#         "role": "user",
#         "content": "Classify the contractual clauses of the following text to one of these options: Adjustments, Agreements, Amendments, Anti-Corruption Laws, Applicable Laws, Approvals, Arbitration, Assignments, Assigns, Authority, Authorizations, Base Salary, Benefits, Binding Effects, Books, Brokers, Capitalization, Change In Control, Closings, Compliance With Laws, Confidentiality, Consent To Jurisdiction, Consents, Construction, Cooperation, Costs, Counterparts, Death, Defined Terms, Definitions, Disability, Disclosures, Duties, Effective Dates, Effectiveness, Employment, Enforceability, Enforcements, Entire Agreements, Erisa, Existence, Expenses, Fees, Financial Statements, Forfeitures, Further Assurances, General, Governing Laws, Headings, Indemnifications, Indemnity, Insurances, Integration, Intellectual Property, Interests, Interpretations, Jurisdictions, Liens, Litigations, Miscellaneous, Modifications, No Conflicts, No Defaults, No Waivers, Non-Disparagement, Notices, Organizations, Participations, Payments, Positions, Powers, Publicity, Qualifications, Records, Releases, Remedies, Representations, Sales, Sanctions, Severability, Solvency, Specific Performance, Submission To Jurisdiction, Subsidiaries, Successors, Survival, Tax Withholdings, Taxes, Terminations, Terms, Titles, Transactions With Affiliates, Use Of Proceeds, Vacations, Venues, Vesting, Waiver Of Jury Trials, Waivers, Warranties, Withholdings.
#                     text: Executive agrees to be employed with the Company, and the Company agrees to employ Executive, during the Term and on the terms and conditions set forth in this Agreement. Executive agrees during the term of this Agreement to devote substantially all of Executive\u2019s business time, efforts, skills and abilities to the performance of Executive\u2019s duties to the Company and to the furtherance of the Company's business."
#     }
# ]
#
# formats.empty  - pass inputs as a single string
#
# "Classify the contractual clauses of the following text to one of these options: Adjustments, Agreements, Amendments, Anti-Corruption Laws, Applicable Laws, Approvals, Arbitration, Assignments, Assigns, Authority, Authorizations, Base Salary, Benefits, Binding Effects, Books, Brokers, Capitalization, Change In Control, Closings, Compliance With Laws, Confidentiality, Consent To Jurisdiction, Consents, Construction, Cooperation, Costs, Counterparts, Death, Defined Terms, Definitions, Disability, Disclosures, Duties, Effective Dates, Effectiveness, Employment, Enforceability, Enforcements, Entire Agreements, Erisa, Existence, Expenses, Fees, Financial Statements, Forfeitures, Further Assurances, General, Governing Laws, Headings, Indemnifications, Indemnity, Insurances, Integration, Intellectual Property, Interests, Interpretations, Jurisdictions, Liens, Litigations, Miscellaneous, Modifications, No Conflicts, No Defaults, No Waivers, Non-Disparagement, Notices, Organizations, Participations, Payments, Positions, Powers, Publicity, Qualifications, Records, Releases, Remedies, Representations, Sales, Sanctions, Severability, Solvency, Specific Performance, Submission To Jurisdiction, Subsidiaries, Successors, Survival, Tax Withholdings, Taxes, Terminations, Terms, Titles, Transactions With Affiliates, Use Of Proceeds, Vacations, Venues, Vesting, Waiver Of Jury Trials, Waivers, Warranties, Withholdings.
# text: Each Credit Party shall maintain, in all material respects, proper, complete and consistent books of record with respect to such Person\u2019s operations, affairs, and financial condition.\u00a0From time to time upon reasonable prior notice, each Credit Party shall permit any Lender, at such reasonable times and intervals and to a reasonable extent and under the reasonable guidance of officers of or employees delegated by officers of such Credit Party, to, subject to any applicable confidentiality considerations, examine and copy the books and records of such Credit Party, to visit and inspect the Property of such Credit Party, and to discuss the business operations and Property of such Credit Party with the officers and directors thereof (provided that, so long as no Event of Default has occurred and is continuing, the Lenders shall be entitled to only one such visit per year coordinated by the Administrative Agent).
# The contractual clauses is Records
#
# text: Executive agrees to be employed with the Company, and the Company agrees to employ Executive, during the Term and on the terms and conditions set forth in this Agreement. Executive agrees during the term of this Agreement to devote substantially all of Executive\u2019s business time, efforts, skills and abilities to the performance of Executive\u2019s duties to the Company and to the furtherance of the Company's business.
# The contractual clauses is "

for model, model_name in model_list:
    print(model_name)
    card = "cards.ledgar"
    template = "templates.classification.multi_class.instruction"
    for format in [
        "formats.chat_api[place_instruction_in_user_turns=True]",
        "formats.chat_api",
        "formats.empty",
    ]:
        for system_prompt in [
            "system_prompts.empty",
        ]:
            if model_name == "wml-gen" and "formats.chat_api" in format:
                continue
            if model_name == "wml-chat" and "formats.chat_api" not in format:
                continue
            dataset = load_dataset(
                card=card,
                format=format,
                system_prompt=system_prompt,
                template=template,
                num_demos=5,
                demos_pool_size=100,
                loader_limit=1000,
                max_test_instances=128,
                split="test",
            )
            type_of_input = type(dataset[0]["source"])

            print("Starting inference...")
            start = time.perf_counter()
            predictions = model(dataset)
            end = time.perf_counter()
            duration = end - start
            print("End of inference...")

            results = evaluate(predictions=predictions, data=dataset)

            print(
                f"Sample input and output for format '{format}' and system prompt '{system_prompt}':"
            )

            print("Example prompt:")

            print(json.dumps(results.instance_scores[0]["source"], indent=4))

            print("Example prediction:")

            print(json.dumps(results.instance_scores[0]["prediction"], indent=4))

            global_scores = results.global_scores
            df.loc[len(df)] = [
                model_name,
                format,
                system_prompt,
                global_scores["score"],
                global_scores["score_ci_low"],
                global_scores["score_ci_high"],
                duration,
                len(predictions),
                type_of_input,
            ]

            df = df.round(decimals=2)
            print(df.to_markdown())
