from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate
from dataclasses import dataclass

@dataclass
class prometheusTemplateThree:
    with_ref:bool
    general_criteria:str
    score_1_desc:str
    score_2_desc:str
    score_3_desc:str
    template_name_suffix:str

    def get_score_range(self)->int:
        return 3

    def get_score_rubrics_section(self)->str:
        return f"{self.general_criteria}\n Score 1: {self.score_1_desc}\n Score 2: {self.score_2_desc}\n Score 3: {self.score_3_desc}\n\n"
    
    def get_template_path(self)->str:
        template_ref_str = "_with_reference" if self.with_ref else ''
        return f"templates.response_assessment.rating.prometheus_single_turn_{self.template_name_suffix}_{self.get_score_range()}{template_ref_str}"
    
    def get_ref_str_for_instruction(self)->str:
        return f" a reference answer that gets a score of {self.get_score_range()}," if self.with_ref else ''

    def get_ref_insertion_str(self)->str:
        title = f"#Reference Answer (Score {self.get_score_range()}):\n"
        ref_insertion_str = title + "{reference_answer}\n\n"
        return ref_insertion_str if self.with_ref else ''



@dataclass
class prometheusTemplateFive(prometheusTemplateThree):
    score_4_desc:str
    score_5_desc:str

    def get_score_range(self) -> int:
        return 5

    def get_score_rubrics_section(self) -> str:
        parent_rubrics = super().get_score_rubrics_section()
        return parent_rubrics.replace("\n\n",f"\n Score 4: {self.score_4_desc}\n Score 5: {self.score_5_desc}\n\n")


p_templates = []
for with_ref in (True,False):
    p_templates.extend([
        prometheusTemplateFive(
            with_ref=with_ref,
            general_criteria="Does the response comply with the instruction accurately and completly?",
            # "Does the response execute the instruction to evaluate correctly, faithfully and completly?"
            score_1_desc="The response does not comply with the instruction or only minimally addresses it, with significant omissions or irrelevant content.",
            score_2_desc="The response addresses some aspects of the instruction but is largely incomplete, inaccurate, or contains irrelevant details",
            score_3_desc="The response complies with parts of the instruction but is incomplete, contains inaccuracies, or includes a lot of irrelevant information",
            score_4_desc="The response mostly complies with the instruction but may lack full detail, have minor inaccuracies, or include some irrelevant information",
            score_5_desc="The response fully complies with the instruction, meeting all requirements accurately and without unnecessary or redundant content",
            template_name_suffix=f"inst_comply"
        ),
        prometheusTemplateThree(
            with_ref=with_ref,
            general_criteria="Does the response comply with the instruction accurately and completly?",
            # "Does the response execute the instruction to evaluate correctly, faithfully and completly?"
            score_1_desc="The response does not comply with the instruction or only minimally addresses it, with significant omissions or irrelevant content.",
            score_2_desc="The response complies with parts of the instruction but is incomplete, contains inaccuracies, or includes a lot of irrelevant information",
            score_3_desc="The response fully complies with the instruction, meeting all requirements accurately and without unnecessary or redundant content",
            template_name_suffix=f"inst_comply"
        ),
        prometheusTemplateThree(
            with_ref=with_ref,
            general_criteria="Does the response adhere to the instruction precisely, without including additional text?",
            score_1_desc="The response does not adhere to the instruction and includes additional text",
            score_2_desc="The response partially follows the instruction or contains additional text",
            score_3_desc="The response adheres precisely to the instruction without including any additional text",
            template_name_suffix=f"inst_strict"
           
        ),
        prometheusTemplateThree(
            with_ref=with_ref,
            general_criteria="Does the response classify the text correctly, adhering to the instruction and outputting classes only?",
            score_1_desc="The response does not classify the text correctly according to the instruction",
            score_2_desc="The response classifies the text correctly but either not according to the instuction or it adds text other than the classification itself",
            score_3_desc="The response classifies the text correctly according to the instruction, adhering strictly to instruction and and without any text other than the class itself",
            template_name_suffix=f"inst_classify"
        )
    ]
    )


for ptemp in p_templates:
    print(f"preparing {ptemp.get_template_path()}...")
    SCORE_RANGE = ptemp.get_score_range()
    add_to_catalog(
        InputOutputTemplate(
            instruction="###Task Description:\n"
            "An instruction (might include an Input inside it), a response to evaluate,"
            f"{ptemp.get_ref_str_for_instruction()} and a score rubric representing a evaluation criteria are given.\n"
            "1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.\n"
            f"2. After writing a feedback, write a score that is an integer between 1 and {SCORE_RANGE} . You should refer to the score rubric.\n"
            f"3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and {SCORE_RANGE})\n"
            "4. Please do not generate any other opening, closing, and explanations.\n\n",
            input_format="###The instruction to evaluate:\n"
            "{question}\n\n"
            "###Response to evaluate:\n"
            "{answer}\n\n"
            f"{ptemp.get_ref_insertion_str()}"
            "###Score Rubrics:\n"
            f"{ptemp.get_score_rubrics_section()}"
            "###Feedback:",
            output_format="[{rating}]",
            postprocessors=[
                f"processors.extract_prometheus_rating_judgment_{SCORE_RANGE}",
            ],
        ),
        ptemp.get_template_path(),
        overwrite=True,
    )
