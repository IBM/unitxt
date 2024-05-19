# Enriches the cards (to begin with) with info scraped from hf.
# Copies the exhastive search for task-card generators - from test_preparation,
# and assumes a tweaking of test_card to read:
# card_info = {"path" : card.loader.path if hasattr(card.loader, "path") else "",
#              "name": card.loader.name if hasattr(card.loader, "name") else "",
#              "tags": card.__tags__ if hasattr(card, "__tags__") else "",
#              "desc": card.__description__ if hasattr(card, "__description__") else ""}
# print(json.dumps(card_info))
# return
#
# Also, comment out every warnings that interferes with the printout.
# metrics.py and task.py has some.
# Tweak 'add_to_catalog' to return right away, so it does not invoke checking
# and warnings.
#
#
import glob
import json
import os
import sys
import warnings
from subprocess import PIPE, Popen

from datasets import load_dataset_builder
from huggingface_hub import dataset_info
from huggingface_hub.repocard import RepoCard

from ..logging_utils import get_logger
from ..text_utils import lines_defining_obj

warnings.filterwarnings("ignore")
# does not help when execution goes out to the task-card generators, and from them on
# to test_card

logger = get_logger()
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
logger.info("project_dir: ", project_dir)
glob_query = os.path.join(project_dir, "prepare", "cards", "co*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
all_preparation_files.sort()

logger.info(f"Annotating files: {all_preparation_files}")
with open("all_infos.txt", "w") as writer:
    writer.write("starting annotations\n")
for file in all_preparation_files:
    logger.info(
        "\n_____________________________________________\n"
        f"  Testing preparation file:\n  {file}."
        "\n_____________________________________________\n"
    )
    if file.endswith("cards/atta_q.py"):
        # crashes the run, for some reason
        continue
    if file.endswith("_ANNOTATED.py"):
        continue
    with open("all_infos.txt", "a") as writer:
        writer.write(f"\nCard: {file}\n")

    # go out to the task-card generator, and catch all printouts
    # invoked from there.
    # we planted one at the beginning of test_card, as explained above
    command = [sys.executable, file]
    process = Popen(command, stdout=PIPE, stdin=PIPE)
    stdout, stderr = process.communicate()
    # print("From Here: \n\n", stdout, "\n\n")
    as_string = str(stdout)
    if "{" not in as_string:
        # not a decent json-dict
        continue
    # print("as string:", as_string)
    # print("as_string[0]", as_string[0])
    # print("as_string[1]", as_string[1])
    # print("as_string[2]", as_string[2])
    # print("as_string[-1]", as_string[-1])
    # print("as_string[-2]", as_string[-2])
    # print("as_string[-3]", as_string[-3])
    # print("as_string[-4]", as_string[-4])
    for_jsons = as_string[2:-3].split("\\n")
    # for for_json in for_jsons:
    #     print(for_json)

    # Even where there are more than one TaskCard in one file (not in a loop, but actually
    # more than one line of code that contains 'TaskCard( )'), we assume all
    # have the same info. (see e.g. coedit.py)
    # We scrape info by the first card, and plant in all of them
    #
    card = json.loads(for_jsons[0])
    if len(card["path"]) == 0:
        # not a LoadHF
        continue
    ds_builder = None
    try:
        if card["name"] is None:
            ds_builder = load_dataset_builder(card["path"])
            card["name"] = ""  # for printouts
        else:
            ds_builder = load_dataset_builder(card["path"], card["name"])
    except Exception:
        with open("all_infos.txt", "a") as writer:
            writer.write(
                "\n!!!!!! tried dataset_builder on path and name but failed -------------\n"
            )
            writer.write(
                "loader path and name: '"
                + card["path"]
                + "' and '"
                + card["name"]
                + "'\n"
            )
            ds_builder = None

    ds_info = None
    try:
        ds_info = dataset_info(repo_id=card["path"])
    except Exception:
        with open("all_infos.txt", "a") as writer:
            writer.write("\n@@@ tried data_set.info on path but failed -------------\n")
            writer.write(
                "loader path and name: '"
                + card["path"]
                + "' and '"
                + card["name"]
                + "'\n"
            )
            ds_info = None

    repo_card = None
    try:
        repo_card = RepoCard.load(card["path"])
    except Exception:
        with open("all_infos.txt", "a") as writer:
            writer.write("\n### tried repocard load on path but failed -------------\n")
            writer.write(
                "loader path and name: '"
                + card["path"]
                + "' and '"
                + card["name"]
                + "'\n"
            )
            repo_card = None

    with open("all_infos.txt", "a") as writer:
        if (
            (ds_builder is not None and len(ds_builder.info.description) > 0)
            or ds_info
            or repo_card
        ):
            writer.write("\n-------------------------------------\n")
            writer.write(
                "loader path and name: '"
                + card["path"]
                + "' and '"
                + card["name"]
                + "'\n"
            )
        if ds_builder is not None and len(ds_builder.info.description) > 0:
            writer.write("ds_builder_all info:\n")
            writer.write(str(ds_builder.info))
            writer.write("\n\nJust Description:\n")
            writer.write(str(ds_builder.info.description))
            writer.write("\nJust Citation:\n")
            writer.write(str(ds_builder.info.citation))
            writer.write("\nJust Homepage:\n")
            writer.write(str(ds_builder.info.homepage))
            writer.write("\nJust Features:\n")
            writer.write(str(ds_builder.info.features))
        if ds_info is not None:
            writer.write("from ds_info: di.tags\n")
            writer.write(str(ds_info.tags) + "\n")
            writer.write("from ds_info: ds_info.card_data\n")
            writer.write(str(ds_info.card_data) + "\n")
            writer.write("ds_info.data.todict:")
            writer.write(
                str(ds_info.card_data.to_dict())
                if ds_info.card_data is not None
                else "None"
            )
            writer.write("\nfrom ds_info: ds_info.description\n")
            writer.write(
                (str(ds_info.description) if hasattr(ds_info, "description") else "")
                + "\n"
            )

        if repo_card is not None:
            writer.write("from repo_card.data.tags:\n")
            writer.write(str(repo_card.data.tags) + "\n")
            writer.write("from repo_card.data:\n")
            writer.write(str(repo_card.data) + "\n")

    tags_to_add = (
        None
        if ds_info is None or ds_info.tags is None or len(ds_info.tags) == 0
        else ds_info.tags
    )
    # scraped_description = None if ds_info is None or ds_info.description is None or len(ds_info.description) == 0 else ds_info.description
    # print("scraped description:")
    # print(scraped_description[0])
    # print(scraped_description[1])
    # print(scraped_description[2])
    # print(scraped_description[3])
    update_to_tags = (
        {"dataset_info_tags": tags_to_add} if tags_to_add is not None else None
    )
    to_lines = []
    to_file = file[:-3] + "_ANNOTATED.py"
    # the name of the file to be generated with the annotations

    with open(file) as fp:
        all_lines = fp.readlines()
    current_line = 0
    while True:
        # locate next definition of TaskCard in the input file
        starting_card, ending_card = lines_defining_obj(
            all_lines=all_lines, obj_name="TaskCard(", start_search_at_line=current_line
        )
        if starting_card == -1:
            # no more task_cards in file
            to_lines.extend(all_lines[current_line:])
            with open(file, "w") as fp2:
                fp2.writelines(to_lines)
            break

        # the indent of the _tags__ tag exceeds by 4 the indent of the closing ) of the taskCard:
        tags_indent = " " * (4 + all_lines[ending_card].index(")"))
        # we have a taskcard to update
        # copy all lines down to start of this taskcard on to the annotated file
        to_lines.extend(all_lines[current_line:starting_card])
        current_line = starting_card

        if update_to_tags is None:
            # nothing to update, copy over to the end of the task_card
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
            continue

        # we have tags to add to this file
        # See if it has tags line already:
        starting_tags_in_card, ending_tags_in_card = lines_defining_obj(
            all_lines=all_lines[starting_card:ending_card], obj_name="__tags__"
        )
        if starting_tags_in_card != -1:
            # there was a __taggs__ field in the existing card, we will update it now
            current_tags = card["tags"]
            current_tags.update(update_to_tags)
            # copy over all the lines down to the existing tag lines
            to_lines.extend(
                all_lines[current_line : starting_card + starting_tags_in_card]
            )
            current_line = starting_card + starting_tags_in_card
            to_lines.append(
                tags_indent + "__tags__ = " + json.dumps(current_tags) + ",\n"
            )
            # skip the input lines that belonged to the existing __tags__
            current_line = starting_card + ending_tags_in_card + 1
            # and copy over all lines down to the end of the card
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
        else:
            # there was no tags in existing card, we copy all lines of cards and then append
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
            to_lines.append(
                tags_indent + "__tags__ = " + json.dumps(update_to_tags) + ",\n"
            )
