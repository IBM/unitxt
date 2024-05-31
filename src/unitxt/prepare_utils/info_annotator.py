# Enriches the cards (to begin with) with info scraped from hf.
# Copies the exhastive search for task-card generators - from test_preparation,
# and assumes a tweaking of test_card to read, right as its first lines:
# card_info = {"path" : card.loader.path if hasattr(card.loader, "path") else "",
#              "name": card.loader.name if hasattr(card.loader, "name") else "",
#              "tags": card.__tags__ if hasattr(card, "__tags__") else "",
#              "desc": card.__description__ if hasattr(card, "__description__") else ""}
# jsoned_card_info = str(json.dumps(card_info))
# print(jsoned_card_info)
# return
#
# if the currently existing tags or description are not to be updated with additional info
# from here, but simply be replaced, then can comment out both lines above
# that return the existing tags and desc.
#
# Also, comment out every warnings that interferes with the printout.
# metrics.py and task.py have some. Alternatively, set:
# settings.default_verbosity = "critical"
# in settings_utils.
#
# Tweak 'add_to_catalog' to just return right away, so it does not invoke checking
# and warnings.
#
#
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from subprocess import PIPE, Popen

from datasets import load_dataset_builder
from huggingface_hub import dataset_info
from huggingface_hub.repocard import RepoCard

from ..logging_utils import get_logger
from ..text_utils import lines_defining_obj_in_card

logger = get_logger()
logger.setLevel(logging.INFO)
# so, with settings.default_verbosity = "critical", only this module catches stdout and stderr
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
logger.info("project_dir: ", project_dir)
glob_query = os.path.join(project_dir, "prepare", "cards", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
all_preparation_files.sort()

logger.info(f"Annotating files: {all_preparation_files}")
# all_infos.txt will gather all the accessible info in all ways tried here
with open("all_infos.txt", "w") as writer:
    writer.write("starting annotations\n")
for file in all_preparation_files:
    logger.info(
        "\n_____________________________________________\n"
        f"  Testing preparation file:\n  {file}."
        "\n_____________________________________________\n"
    )
    with open("all_infos.txt", "a") as writer:
        writer.write(f"\nCard file: {file}\n")

    # go out to the task-card generator, and catch all printouts
    # invoked from there.
    # we planted one at the beginning of test_card, as explained above
    command = [sys.executable, file]
    process = Popen(command, stdout=PIPE, stdin=PIPE)
    stdout, stderr = process.communicate()
    as_string = str(stdout)
    if "{" not in as_string:
        # not a decent json-dict
        continue
    for_jsons = as_string[2:-3].split("\\n")
    # more than one line arrive when more than once is test_card(card) invoked from the file tested,
    # e.g. coedit.py
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
            writer.write("\n--------------scraped info-----------------------\n")
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

    scraped_description = (
        None
        if ds_info is None
        or not hasattr(ds_info, "description")
        or ds_info.description is None
        or len(ds_info.description) == 0
        else str(ds_info.description)
    )
    if scraped_description is not None:
        with open("all_infos.txt", "a") as writer:
            scraped_description = scraped_description.strip()
            writer.write("scraped description from ds_info:\n")
            writer.write(scraped_description)
            writer.write('\ntranslated s+ and escape " \n')
            scraped_description = re.sub(r"[\t\n]+", "\n", scraped_description)
            scraped_description = scraped_description.strip()
            scraped_description = scraped_description.replace('"', '\\"')
            writer.write(scraped_description)

    tags_to_add = (
        []
        if ds_info is None or ds_info.tags is None or len(ds_info.tags) == 0
        else ds_info.tags
    )
    # this is a list of strings. Some of them have : inside them and some do not.
    # we make them all one dict, the keys are the (set of the) left side of :  or the only string if : is not inside.
    # the values are all the right sides of the : where if some left side shows more than once, then we gather into a list
    # all its right sides,  and for the strings without :  we set the right side to be True.
    # the following code replaces the existing (if exists) __tags__ by  tags_to_add. if existing tags are to be updated
    # (not replace) then the existing tags, coming above from stdout, should be combined with these
    # coming from ds_info, and the result is to replace the existing __tags__. (by the following code)

    the_dict_to_become_tags = defaultdict(list)
    for tag in tags_to_add:
        if ":" in tag:
            the_dict_to_become_tags[tag[: tag.index(":")].strip()].append(
                tag[1 + tag.index(":") :].strip()
            )
        else:
            the_dict_to_become_tags["singletons"].append(tag.strip())

    for key in the_dict_to_become_tags:
        if (
            isinstance(the_dict_to_become_tags[key], list)
            and (len(the_dict_to_become_tags[key]) == 1)
            and (key != "singletons")
        ):
            the_dict_to_become_tags[key] = the_dict_to_become_tags[key][0]
    if the_dict_to_become_tags is None or len(the_dict_to_become_tags) == 0:
        string_for_tags = None
    else:
        string_for_tags = json.dumps(the_dict_to_become_tags, sort_keys=True)
        string_for_tags = string_for_tags.replace(": true,", ": True,").replace(
            ": true}", ": True}"
        )

    to_lines = []
    with open(file) as fp:
        all_lines = fp.readlines()
    current_line = 0
    while True:
        # locate next definition of TaskCard in the input file
        starting_card, ending_card = lines_defining_obj_in_card(
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

        # this piece of cocde adds description,  for adding __tags__, replace by string_for_tags
        if scraped_description is None:
            # nothing to update, copy over to the end of the task_card
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
            continue

        # we have description to add to this file
        # for tags, this whole preprocessing is not needed, we have one string to add,
        # it is ready to add, showing how in the sequel
        split_description = scraped_description.split("\n")
        for i in range(len(split_description)):
            split_description[i] = split_description[i].strip()
        for i in range(len(split_description)):
            split_description[i] = split_description[i] + "\\n"
        for i in range(len(split_description)):
            split_description[i] = '"' + split_description[i] + '"\n'
        for i in range(len(split_description)):
            split_description[i] = tags_indent + "    " + split_description[i]

        # See if it has description line already:
        starting_tags_in_card, ending_tags_in_card = lines_defining_obj_in_card(
            all_lines=all_lines[starting_card:ending_card],
            obj_name="__description__",
            # obj_name="__tags__",
        )
        if starting_tags_in_card != -1:
            # there was a __tags__ field in the existing card, we will replace it now by
            # the dict_to_become_tags. if we need to update - we will do it here.
            # current_tags = card["tags"]
            # current_tags.update(update_to_tags)
            # copy over all the lines down to the existing tag lines
            to_lines.extend(
                all_lines[current_line : starting_card + starting_tags_in_card]
            )
            # the following piece is for __description__
            current_line = starting_card + starting_tags_in_card
            to_lines.append(tags_indent + "__description__=(")
            to_lines.extend(split_description)
            to_lines.append(tags_indent + "),\n")
            # in case of __tags__: simply use, instead of the above three lines:
            # to_lines.append(tags_indent + "__tags__=" + string_for_tags + ",\n")

            # skip the input lines that belonged to the existing __tags__
            current_line = starting_card + ending_tags_in_card + 1
            # and copy over all lines down to the end of the card
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
        else:
            # there was no tags in existing card, we copy all lines of cards
            # and then append the lines of the new __description__
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
            to_lines.append(tags_indent + "__description__=(")
            to_lines.extend(split_description)
            to_lines.append(tags_indent + "),\n")
            # in case of __tags__: simply use, instead of the above three lines:
            # to_lines.append(tags_indent + "__tags__ = " + string_for_tags + ",\n")
