# Enriches the cards, whose loader= LoadHF (to begin with) with split-names scraped from hf.
# Copies the exhastive search for task-card generators - from test_preparation,
# and assumes a tweaking of test_card to read, right as its first lines:
# card_info = {"path" : card.loader.path if hasattr(card.loader, "path") else "",
#              "name": card.loader.name if hasattr(card.loader, "name") else ""}
# jsoned_card_info = str(json.dumps(card_info))
# print(jsoned_card_info)
# return
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
import sys
from subprocess import PIPE, Popen

from datasets import get_dataset_split_names
from unitxt.logging_utils import get_logger
from unitxt.text_utils import lines_defining_obj_in_card

logger = get_logger()
logger.setLevel(logging.INFO)
# so, with settings.default_verbosity = "critical", only this module catches stdout and stderr
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
logger.info("project_dir: ", project_dir)
glob_query = os.path.join(project_dir, "prepare/cards", "**", "*.py")
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
    if "path" not in card or card["path"] is None or len(card["path"]) == 0:
        # not a decent LoadHF
        continue
    split_names = []
    try:
        split_names = get_dataset_split_names(
            path=card["path"], config_name=card["name"], trust_remote_code=True
        )
        if card["name"] is None:
            card["name"] = ""  # for printouts

    except Exception:
        if card["name"] is None:
            card["name"] = ""  # for printouts

        with open("all_infos.txt", "a") as writer:
            writer.write(
                "\n!!!!!! tried get_dataset_split_names on path and name but failed -------------\n"
            )
            writer.write(
                f"LoadHF of path '{card['path']}' and name '{card['name']}' failed to return split names\n"
            )
        continue  # to next prepare file

    with open("all_infos.txt", "a") as writer:
        if split_names is not None and len(split_names) > 0:
            writer.write("\n--------------scraped info-----------------------\n")
            writer.write(
                f"LoadHF of path '{card['path']}' and name '{card['name']}' leads to splits {split_names}\n"
            )

    if split_names is None or len(split_names) == 0:
        continue  # to next prepare file

    to_lines = []
    with open(file) as fp:
        all_lines = fp.readlines()
    current_line = 0
    while True:
        # locate next definition of LoadHF in the input file
        starting_card, ending_card = lines_defining_obj_in_card(
            all_lines=all_lines, obj_name="LoadHF(", start_search_at_line=current_line
        )
        if starting_card == -1:
            # no more LoadHF in file
            to_lines.extend(all_lines[current_line:])
            with open(file, "w") as fp2:
                fp2.writelines(to_lines)
            break

        # we have a LoadHF to update
        # copy all lines down to (not including) last line of this LoadHF over to the annotated file
        to_lines.extend(all_lines[current_line:ending_card])
        last_line_of_loadhf = all_lines[ending_card]
        if starting_card == ending_card:
            # one liner, need to push comma and the splits
            major_closing = last_line_of_loadhf.rfind(")")
            new_last_line = (
                last_line_of_loadhf[:major_closing]
                + f", all_splits={split_names}"
                + last_line_of_loadhf[major_closing:]
            )
            to_lines.append(new_last_line)
        else:
            # multi liner LoadHF, add split as one more line
            if not to_lines[-1][-2] == ",":
                to_lines[-1] = to_lines[-1][:-1] + ",\n"
            indent = 4 + all_lines[starting_card].index("loader")
            prepend = " " * indent
            to_lines.append(prepend + f"all_splits={split_names},\n")
            to_lines.append(last_line_of_loadhf)

        current_line = ending_card + 1
