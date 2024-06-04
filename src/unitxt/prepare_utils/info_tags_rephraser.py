# rephrase the contents of the __tags__ tag in card generators
#
#
import glob
import json
import logging
import os
import re
from collections import defaultdict

from ..logging_utils import get_logger
from ..text_utils import lines_defining_obj_in_card

logger = get_logger()
logger.setLevel(logging.INFO)
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
logger.info("project_dir: ", project_dir)
glob_query = os.path.join(project_dir, "prepare", "cards", "*.py")
all_preparation_files = glob.glob(glob_query, recursive=True)
all_preparation_files.sort()

logger.info(f"Annotating files: {all_preparation_files}")
# all_infos.txt will gather all the accessible info in all ways tried here
for file in all_preparation_files:
    logger.info(
        "\n_____________________________________________\n"
        f"  Testing preparation file:\n  {file}."
        "\n_____________________________________________\n"
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

        starting_tags_in_card, ending_tags_in_card = lines_defining_obj_in_card(
            # all_lines=all_lines[starting_card:ending_card], obj_name="__description__"
            all_lines=all_lines[starting_card:ending_card],
            obj_name="__tags__",
        )
        if starting_tags_in_card == -1:
            # no __tags__ in card, nothing to rephrase:
            to_lines.extend(all_lines[current_line:ending_card])
            current_line = ending_card
            continue  # to next occurrence of TaskCard( in file

        # there were __tags__ in this card, rephrase them:
        the_dict_to_become_tags = defaultdict(list)
        current_tags_line = "".join(
            all_lines[
                starting_card + starting_tags_in_card : starting_card
                + ending_tags_in_card
                + 1
            ]
        )
        current_tags_line = (
            current_tags_line.replace("__ =", "__=")
            .replace("= {", "={")
            .replace("\n", "")
        )
        assert current_tags_line.startswith(tags_indent + "__tags__=")
        assert current_tags_line.endswith(",")
        current_tags_line = current_tags_line[len(tags_indent) + 9 : -1]
        current_tags_line = current_tags_line.replace(": True,", ": true,").replace(
            ": True}", ": true}"
        )
        current_tags_line = re.sub(
            pattern=r",[ ]+}", repl="}", string=current_tags_line
        )
        current_tags_line = re.sub(
            pattern=r",[ ]+]", repl="]", string=current_tags_line
        )
        tags_to_add = json.loads(current_tags_line)
        for key, val in tags_to_add.items():
            if key == "singletons":
                the_dict_to_become_tags["flags"] = val
            else:
                the_dict_to_become_tags[key] = val
        if the_dict_to_become_tags is None or len(the_dict_to_become_tags) == 0:
            string_for_tags = None
        else:
            string_for_tags = json.dumps(the_dict_to_become_tags, sort_keys=True)
            string_for_tags = string_for_tags.replace(": true,", ": True,").replace(
                ": true}", ": True}"
            )
        to_lines.extend(all_lines[current_line : starting_card + starting_tags_in_card])
        to_lines.append(tags_indent + "__tags__=" + string_for_tags + ",\n")

        # skip the input lines that belonged to the existing __tags__
        current_line = starting_card + ending_tags_in_card + 1
        # and copy over all lines down to the end of the card
        to_lines.extend(all_lines[current_line:ending_card])
        current_line = ending_card
