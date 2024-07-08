from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    IbmGenAiInferenceEngineParams,
    IbmGenAiInferenceEngine,
)
from unitxt.loaders import LoadCSV
from unitxt.catalog import add_to_catalog
from unitxt.operators import IndexOf, RenameFields, Set
from unitxt.test_utils.card import test_card
from unitxt.text_utils import print_dict
from unitxt.operators import RenameFields, Set
from unitxt.blocks import (
    Task,
    TaskCard,
    TemplatesList,
    InputOutputTemplate,
)
import time,random
import pandas as pd

logger = get_logger()

card = TaskCard(
    loader=LoadCSV(
        files={"test": "/Users/michalshmueli-scheuer/Downloads/sap_conversation_summ.csv"},
    ),
    preprocess_steps=[
        RenameFields(field_to_field={"document": "document"}),
        Set(fields={"summary": "dummy summary", "document_type": "text"}),
    ],
    task="tasks.summarization.abstractive[metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mt_bench_single_turn]]",
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{document}\n",
                output_format="{summary}",
                instruction="You are an expert service agent who specializes in summarizing SAP issues. Your goal is to create a comprehensive summary using only information provided in the context, which is the last section.\n" 
                            "Please read and understand all sections below. It will be extremely important to summarize the context correctly.\n"
                            "Category Definitions\n"
                            "Definition of individual categories which are useful to extract relevant information from the context.\n"
                            "1.Escalation: Details of issue escalation along with the Escalation ID, if available.\n"
                            "2.Symptom: Description of the exact issue faced and criticality of the issue.\n"
                            "3.Environment: Name of environments, systems and product versions etc which have this issue.\n"
                            "4.Steps to Reproduce: Details of steps followed by the agent to reproduce/recreate the issue.\n"
                            "5.Troubleshooting Steps: Details of steps followed by the agent to identify main reason for the issue. Also include steps followed to resolve the issue.\n"
                            "6.Solution Proposed: Details of final suggestions/steps proposed to resolve the issue.\n"
                            "7.Cause of the issue: Details of reasons which caused the issue.\n"
                            "8.Additional information: List of all Knowledge Articles, SAP notes, SAP Help articles, Incident IDs, Escalation IDs and Service Requests.\n\n"
                            "Example Summary Output:\n"
                            "Please do not use the content of this example as this is just a dummy example to outline summary format.\n"
                                "1.Escalation:\n"
                                   "-There is an open escalation for this case: ESC0492771.\n"
                                "2.Symptom:\n"
                                    "-The customer reported a Solr timeout error while running a full index on their SAP Commerce Cloud.\n"
                                    "-The issue started occurring in the last 24 hours and was blocking business validations in the environment.\n"
                                "3.Environment:\n"
                                    "-Source: portal\n"
                                    "-Type: Test System\n"
                                    "-System ID/Name: CLOUD / cs2wpi1y8e-carhartti1-s1\n"
                                    "-Product version: SAP Commerce Cloud 2005.24\n"
                                    "-System managed by: SAP Cloud Tenant- Install base item: cs2wpi1y8e-carhartti1-s\n"
                                    "-Cloud Portal: https://placeholderlink.com\n"
                                    "-Automation Engine: https://placeholderlink.com\n"
                                    "-Sold product: HYBRIS.\n"
                                "4.Steps to Reproduce:\n"
                                    "-From Backoffice, navigate to Facet Search Configuration.\n"
                                    "-Select an index configuration.\n"
                                    "-Run Index to Full.\n"
                                "5.Troubleshooting Steps:\n"
                                    "-The SAP Product Support team conducted an investigation, collecting data and conducting research (KB0017381, KB0216593).\n"
                                    "-The solution provided was to manually remove these remnant cores from Solr pods and restart all zookeeper and solr pods.\n"
                                    "-The customer was informed about the analysis and proposed solution. They were also advised to upgrade their Solr to at least version 8.9 to avoid this and other issues that have been resolved in previous versions.\n"
                                    "-The customer confirmed that they were able to successfully execute a full index after the proposed solution was implemented.\n"
                                "6.Solution Proposed:\n"
                                    "-The solution provided was to manually remove these remnant cores from Solr pods and restart all zookeeper and solr pods.\n"
                                "7.Cause of the issue:\n"
                                    "-They identified that the socket timeout was happening due to remnant cores from previous indexes present on Solr pods.\n"
                                "8.Additional information:\n"
                                    "-Knowledge Article: KB0017381\n"
                                    "-SAP Notes: 0001234563\n"
                                    "-SAP Help Articles: Speed Up Processing Incidents\n"
                                    "-Incident ID: INC1234566\n"
                                    "-Escalation ID: ESC123456\n"
                                    "-Service Request: RITM01223456.\n\n"
                            "Important points to consider while summarizing:\n"
                                "-Please make sure the output follows this format - 1.Escalation, 2.Symptom, 3.Environment, 4.Steps to Reproduce, 5.Troubleshooting Steps, 6.Solution Proposed, 7.Cause of the issue, 8.Additional information.\n"
                                "-An Escalation ID always starts with ESC.\n"
                                "-Please keep the hyperlinks in their original form.\n"
                                "-The placeholders are formatted with angle brackets and colon-separated strings like '&lt;&lt;person&gt;:68cc155d-eb76-4eae-ab49-95bc42ab1f77&gt;' and '&lt;&lt;email&gt;:9bd5fd4c-32c3-4140-a2d1-095dcf6dc036&gt;' which are common ways to represent placeholders or variables in a text-based context. Please always use the placeholders like in the example and never break or change the placeholder format.\n"
                                "-An Incident is a technical term. Please use the word incident only when it is actually used in the context. An Incident ID always starts with INC.\n\n"
                            "Please carefully create a comprehensive summary, category by category, only based on information provided in the context. Please take this work seriously as it is extremely important for my career. "
            ),
        ]
    ),
)

test_card(card, strict=False)
add_to_catalog(card, f"cards.sap_summarization", overwrite=True)


