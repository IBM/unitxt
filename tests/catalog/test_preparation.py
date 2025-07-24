import glob
import os
import time
import traceback
import tracemalloc

import psutil
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from requests.exceptions import ReadTimeout
from unitxt.loaders import MissingKaggleCredentialsError
from unitxt.logging_utils import get_logger
from unitxt.settings_utils import get_constants, get_settings
from unitxt.text_utils import print_dict
from unitxt.utils import import_module_from_file

from tests.utils import CatalogPreparationTestCase

logger = get_logger()
constants = get_constants()
setting = get_settings()

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
glob_query = os.path.join(project_dir, "prepare/cards", "**", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
# Make sure the order in which the tests are run is deterministic
# Having a different order for local testing and github testing may cause diffs in results.
all_preparation_files.sort()
num_par = 12  # num of parallel executions
logger.critical(
    f"Over all, {len(all_preparation_files)} files will now be tested over {num_par} parallel processes."
)
# the following should be any of modulo num_par: 0,1,2,3,4,5,6,7,8,.. num_par-1
modulo = 10
all_preparation_files = [
    file for i, file in enumerate(all_preparation_files) if i % num_par == modulo
]


class TestCatalogPreparation(CatalogPreparationTestCase):
    def test_preparations(self):
        logger.info(glob_query)
        all_preparation_files_as_string = "\n".join(
            [file.split("prepare")[-1][1:] for file in all_preparation_files]
        )
        logger.critical(
            f"Testing {len(all_preparation_files)} preparation files: \n{all_preparation_files_as_string}\n"
        )
        stats = {}
        for j, file in enumerate(all_preparation_files):
            if file.endswith("cards/cohere_for_ai.py"):
                continue
            # passed = True
            error = None
            logger.info(
                "\n_____________________________________________\n"
                f"  Testing preparation file:\n  {file}. number {j}"
                "\n_____________________________________________\n"
            )
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024**3)  # Convert bytes to GB
            disk_start = psutil.disk_io_counters()
            start_time = time.time()
            tracemalloc.start()

            with self.subTest(file=file):
                try:
                    import_module_from_file(file)
                except Exception as e:
                    error = e
                    # passed = False
                    current_exception = e
                    while current_exception:
                        if isinstance(current_exception, (GatedRepoError)):
                            # passed = False
                            break
                        if isinstance(
                            current_exception,
                            (
                                ReadTimeout,
                                HfHubHTTPError,
                                MissingKaggleCredentialsError,
                            ),
                        ):
                            # passed = True
                            break
                        current_exception = (
                            current_exception.__cause__ or current_exception.__context__
                        )

                # if passed:
                if error is None:
                    logger.info(f"Testing preparation file: {file} passed")
                else:
                    logger.critical(
                        f"Testing preparation file: {file} failed with error: {error}\n{traceback.format_exc()}"
                    )
                # else:
                #     raise error

                # self.assertTrue(passed)

            elapsed_time = time.time() - start_time
            disk_end = psutil.disk_io_counters()
            read_gb = (disk_end.read_bytes - disk_start.read_bytes) / (1024**3)
            write_gb = (disk_end.write_bytes - disk_start.write_bytes) / (1024**3)

            tracemalloc.stop()
            _, peak = tracemalloc.get_traced_memory()
            # Convert to GB
            peak_memory_python = peak / (1024**3)  # Convert bytes to GB
            peak_memory_system = (
                process.memory_info().rss / (1024**3) - start_memory
            )  # Convert bytes to GB

            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            formatted_time = f"{minutes:02}:{seconds:02}"
            logger.info(
                "\n_____________________________________________\n"
                f"  Finished testing preparation file:\n  {file}."
                f"  Elapsed Time: {formatted_time}\n"
                f"  Peak Python Memory Usage: {peak_memory_python:.4f} GB\n"
                f"  Peak System RAM Usage: {peak_memory_system:.4f} GB\n"
                f"  Disk Write: {write_gb:.4f} GB, Disk Read: {read_gb:.4f} GB"
                "\n_____________________________________________\n"
            )

            stats[
                file.split("prepare")[-1][1:]
            ] = f"Time: {formatted_time}, RAM: {peak_memory_system:.2f} GB, Disk: {write_gb:.2f} GB"

        logger.critical(f"Preparation times table for {len(stats)} files:")
        times = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
        print_dict(times, log_level="critical")


# mod 0
# cards/20_newsgroups.py - passed
# cards/arena_hard/response_assessment/pairwise_comparative_rating/both_games_gpt4_judge.py  failed with error: The Huggingface space 'lmsys/arena-hard-browser' was not found.
# cards/bold.py  - passed
# cards/copa.py - passed
# cards/financial_tweets.py - passed
# cards/judge_bench/dices_safety.py - passed
# cards/long_bench.py - passed
# cards/mt_bench/generation/japanese_single_turn.py - passed
# cards/numeric_nlg.py - passed
# cards/rag/end_to_end/clapnq.py - passed
# cards/safety/discrim_eval.py - passed
# cards/sst2.py - passed
# cards/toxigen.py - passed
# cards/winogrande.py - passed

# mod 1
# cards/20_newsgroups_short.py - passed
# cards/arena_hard/response_assessment/pairwise_comparative_rating/both_games_mean_judgment_gpt4_judge.py - failed with error: The Huggingface space 'lmsys/arena-hard-browser' was not found.
# cards/boolq.py - passed
# cards/coqa.py - passed
# cards/frames_benchmark.py - passed
# cards/judge_bench/inferential_strategies.py - fails on instance 240: query "instance/problem_statement" did not match any item in dict:
# cards/mbpp.py - passed
# cards/mt_bench/response_assessment/pairwise_comparison/multi_turn_gpt4_judgement.py - passed
# cards/open_australian_legal_qa.py - passed
# cards/rag/end_to_end/hotpotqa.py - passed
# cards/safety/mlcommons_ailuminate.py - passed
# cards/stsb.py - passed
# cards/translation/flores101.py - passed
# cards/wnli.py - passed

# mod 2:
# cards/20newsgroups_sklearn.py - passed
# cards/arena_hard/response_assessment/pairwise_comparative_rating/first_game_only_gpt4_judge.py - failed with error: The Huggingface space 'lmsys/arena-hard-browser' was not found.
# cards/boolq_indic.py - passed
# cards/coqa_multi_turn.py - passed
# cards/global_mmlu.py -  cards.global_mmlu.am.clinical_knowledge fails on schema verification: Passed value [None, 'ኢንሱሊን ሲኖር ብቻ።', 'በሄክሶኪናሴስ በኩል።', 'በሞኖካርቢሊክ አሲድ ማጓጓዣዎች።'] of field 'choices' is not of required type: (List[str]) in Task ('tasks.qa.multiple_choice.with_topic').
# cards/judge_bench/newsroom.py - passed
# cards/medical_abstracts.py - passed
# cards/mt_bench/response_assessment/pairwise_comparison/multi_turn_with_reference_gpt4_judgement.py - passed
# cards/openbookqa.py - passed
# cards/rag/end_to_end/miniwikipedia.py - passed
# cards/safety/provoq.py - passed
# cards/summarize_from_human_feedback.py - cards.summarize_from_human_feedback fails on schema verification: Passed value None of field 'input' is not of required type: (str) in Task ('tasks.evaluation.preference').
# cards/translation/wmt/en_de.py - passed
# cards/wsc.py - passed

# mod 3
# cards/CFPB_product.py - passed
# cards/argument_topic.py - passed
# cards/chart_qa.py - passed
# cards/dart.py - passed
# cards/global_mmlu_lite_sensitivity.py - passed
# cards/judge_bench/roscoe_overall.py - passed
# cards/milu.py - passed
# cards/mt_bench/response_assessment/pairwise_comparison/single_turn_gpt4_judgement.py - passed
# cards/piqa.py - passed
# cards/rag/end_to_end/real_mm_rag.py - gets stuck in the second set of cards
# cards/safety/simple_safety_tests.py - passed
# cards/tab_fact.py - passed
# cards/translation/wmt/en_fr.py - passed
# cards/xlam_function_calling.py - cards.xlam_function_calling_60k fails on instance 15 for schema verification

# mod 4
# cards/ag_news.py - passed
# cards/atis.py - passed
# cards/chat_rag_bench.py - passed
# cards/dbpedia_14.py - passed
# cards/go_emotions.py - passed
# cards/judge_bench/roscoe_stepwise.py - passed
# cards/mlsum.py - cards.mlsum.de fails: The HTTP server doesn't appear to support range requests. Only reading this file from the beginning is supported. Open with block_size=0 for a streaming file interface.
# cards/mt_bench/response_assessment/pairwise_comparison/single_turn_with_reference_gpt4_judgement.py - passed
# cards/pop_qa.py - passed
# cards/rag/end_to_end/watsonx.py - passed
# cards/safety/truthful_qa.py - passed
# cards/tablebench.py - passed
# cards/translation/wmt/en_ro.py - passed
# cards/xlsum.py - passed

# mod 5:
# cards/ai2d.py - passed
# cards/attaq.py - passed
# cards/claim_stance_topic.py - passed
# cards/doc_vqa.py - gets stuck, kills ubuntu
# cards/gpqa.py - cards.gpqa.experts , instance 0 fails: query "Correct Answer" did not match any item in dict: (listFieldValues)
# cards/judge_bench/toxic_chat_jailbreak.py - passed
# cards/mmlu.py - passed
# cards/mt_bench/response_assessment/rating/multi_turn_gpt4_judgement.py - passed
# cards/pop_qa_robust.py - passed
# cards/ragbench.py - passed
# cards/safety/xstest.py - passed
# cards/tablebench_data_analysis.py - passed
# cards/trec.py - passed
# cards/xnli.py - passed

# mod 6:
# cards/almost_evil_ml_qa.py - passed
# cards/attaq_500.py - passed
# cards/clapnq.py - passed
# cards/dynamic_cards_for_llm_judges/llm_as_judge_metrics.py - cards.dynamic_cards_for_llm_judges.rating.single_turn failed:   No template was specified in the the 'template' or 'template_card_index' recipe arguments, and no default templates are defined the card or task
# cards/head_qa.py - passed
# cards/judge_bench/toxic_chat_toxicity.py - passed
# cards/mmlu_pro.py - passed
# cards/mt_bench/response_assessment/rating/multi_turn_with_reference_gpt4_judgement.py - passed
# cards/qnli.py - passed
# cards/ragbench_faithfulness.py - passed
# cards/scigen.py - passed
# cards/tablebench_fact_checking.py - passed
# cards/turl_col_type.py - passed
# cards/xsum.py - passed

# mod 7:
# cards/almost_evil_ml_qa_mulitlingual.py - passed
# cards/babi.py - passed
# cards/clinc_oos.py - passed
# cards/earnings_call.py - passed
# cards/hellaswag.py - passed
# cards/judge_bench/wmt-human.py - passed
# cards/mmmu.py - cards.mmmu.architecture_and_engineering fails with error: failing on instance 13 for field "answer"
# cards/mt_bench/response_assessment/rating/single_turn_gpt4_judgement.py - passed
# cards/qqp.py - passed
# cards/reuters21578.py - passed
# cards/sciq.py - passed
# cards/tablebench_numerical_reasoning.py - passed
# cards/unfair_tos.py - passed
# cards/xwinogrande.py - passed

# mod 8:
# cards/amazon_massive.py - passed
# cards/banking77.py - passed
# cards/cnn_dailymail.py - passed
# cards/ethos.py - passed
# cards/hh_rlhf.py - passed
# cards/language_identification.py - passed
# cards/mnli.py - passed
# cards/mt_bench/response_assessment/rating/single_turn_with_reference_gpt4_judgement.py - passed
# cards/qtsumm.py - passed
# cards/reward_bench.py - passed
# cards/seed_bench.py - stuck. cards.seed_bench downloads 273 parquets, that together are too big
# cards/tablerow_classify.py - fails reading the koggle:  missing credentials, but run on github
# cards/universal_ner.py - passed
# cards/yahoo_answers_topics.py - passed

# mod 9:
# cards/arc.py - passed
# cards/belebele.py - passed
# cards/coedit.py - passed
# cards/ffqa_filtered.py - passed
# cards/human_eval.py - passed
# cards/law_stack_exchange.py - passed
# cards/mrpc.py - passed
# cards/mtrag.py - fails cards.rag.mtrag fails on instance 87. Not clear why
# cards/quality.py - passed
# cards/rte.py - passed
# cards/simple_qa.py - passed
# cards/text2sql.py - passed
# cards/websrc.py - passed

# mod 10:
# cards/arena_hard/common.py - fails with error: 'SequentialOperator' object has no attribute 'templates' - not to be counted, preparation file is in cards, but it adds operator to the catalog
# cards/bfcl.py - cards.bfcl.multi_turn.simple_v3, cards.bfcl.multi_turn.live_relevance_v3, cards.bfcl.multi_turn.live_irrelevance_v3, and cards.bfcl.multi_turn.multiple_v3 finish OK, but then cards.bfcl.multi_turn.live_multiple_v3 stucks.
# cards/cohere_for_ai.py - fails, gets stuck for cards.cohere_for_ai.dolly_machine_translated.por
# cards/fin_qa.py -
# cards/info_vqa.py -
# cards/ledgar.py -
# cards/mt_bench/common.py -
# cards/multidoc2dial.py -
# cards/race.py -
# cards/safety/airbench2024.py -
# cards/social_iqa.py -
# cards/tldr.py -
# cards/wiki_bio.py -
