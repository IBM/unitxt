import argparse
import os.path
import pickle

import torch
from tqdm import tqdm
from unitxt import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    OpenAiInferenceEngine,
)
from unitxt.settings_utils import get_settings
from unitxt.standard import StandardRecipe
from unitxt.struct_data_operators import (
    SerializeTableAsDFLoader,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

settings = get_settings()

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=str, required=True)
parser.add_argument("-num_demos", "--num_demos", type=int, required=True)
parser.add_argument("-out_path", "--out_path", type=str, required=True)
parser.add_argument("-debug", "--debug", type=bool, default=False)
parser.add_argument(
    "-cards",
    "--cards",
    type=str,
    required=True,
)
parser.add_argument(
    "-seeds", "--seeds", type=str, required=False, default=str(settings.seed)
)
parser.add_argument(
    "-shuffle_rows", "--shuffle_rows", type=bool, required=False, default=False
)
parser.add_argument(
    "-shuffle_cols", "--shuffle_cols", type=bool, required=False, default=False
)
args = parser.parse_args()
model_name = args.model
num_demos = args.num_demos
out_path = os.path.join(args.out_path, args.model.split("/")[-1])
debug = args.debug
seeds = args.seeds
shuffle_rows = args.shuffle_rows
shuffle_cols = args.shuffle_cols
cards = args.cards

DEMOS_POOL_SIZE = 10
cards_parsed = cards.split(",")
try:
    seeds_parsed = [int(i) for i in seeds.split(",")]
except:
    seeds_parsed = [settings.seed]
subsets = {}


format = "formats.empty"
if "llama" in model_name:
    format = "formats.llama3_instruct_all_demos_in_one_turn_without_system_prompt"
elif "mixtral" in model_name:
    format = "formats.models.mistral.instruction.all_demos_in_one_turn"


for seed in seeds_parsed:
    for card in cards_parsed:
        subset = {
            card
            + "__json"
            + ("__seed=" + str(seed) if seed != settings.seed else ""): StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=SerializeTableAsJson(shuffle_rows=shuffle_rows, seed=seed),
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
            ),
            card
            + "__markdown"
            + ("__seed=" + str(seed) if seed != settings.seed else ""): StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=SerializeTableAsMarkdown(
                    shuffle_rows=shuffle_rows, seed=seed
                ),
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
            ),
            card
            + "__row_indexed_major"
            + ("__seed=" + str(seed) if seed != settings.seed else ""): StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=SerializeTableAsIndexedRowMajor(
                    shuffle_rows=shuffle_rows, seed=seed
                ),
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
            ),
            card
            + "__df"
            + ("__seed=" + str(seed) if seed != settings.seed else ""): StandardRecipe(
                card="cards." + card,
                template_card_index=0,
                serializer=SerializeTableAsDFLoader(
                    shuffle_rows=shuffle_rows, seed=seed
                ),
                num_demos=num_demos,
                demos_pool_size=DEMOS_POOL_SIZE,
                format=format,
            ),
        }
        subsets.update(subset)


for subset_name, subset in tqdm(subsets.items()):
    # print(
    #     "Running:",
    #     subset_name,
    #     "|",
    #     [f"{arg}: {value} | " for arg, value in vars(args).items()],
    # )

    benchmark = Benchmark(
        max_samples_per_subset=100 if not debug else 5,
        loader_limit=500 if not debug else 100,
        subsets={subset_name: subset},
    )

    test_dataset = list(benchmark()["test"])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    try:
        torch.cuda.empty_cache()
        if "gpt" in model_name:
            inference_model = OpenAiInferenceEngine(
                model_name=model_name,
                max_tokens=100,
                temperature=0.05,
            )
        else:
            inference_model = IbmGenAiInferenceEngine(
                model_name=model_name,
                max_new_tokens=100,
                temperature=0.05,
                # batch_size=16,
            )

            # inference_model = HFPipelineBasedInferenceEngine(
            #     model_name=model_name,
            #     max_new_tokens=100,
            #     use_fp16=True,
            #     # temperature=0 is hard coded in HFPipelineBasedInferenceEngine since it is not allowed to be a param
            # )

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        # exp_name = "__{0}".format(
        #     ("__shuffle_rows" if shuffle_rows else ""),
        # )

        out_file_name = (
            model_name.split("/")[-1]
            + "#"
            + subset_name
            # + (exp_name if exp_name else "")
            + ("__shuffle_rows" if shuffle_rows else "")
            # + "#"
            # + str(datetime.datetime.now())
            + ("_DEBUG" if debug else "")
        )
        curr_out_path = os.path.join(out_path, out_file_name) + ".pkl"
        with open(curr_out_path, "wb") as f:
            pickle.dump(evaluated_dataset, f)
            # print("saved file path: ", curr_out_path)
    except Exception:
        # print(e)
        pass
