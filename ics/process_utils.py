import copy
import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def save_first_n_lines_jsonl(input_path, output_path, n_lines):
    """Reads the first N lines from a JSON Lines file and saves them to a new file.

    Args:
        input_path (str): Path to the input .jsonl file.
        output_path (str): Path to save the output .jsonl file.
        n_lines (int): The number of lines (JSON objects) to keep.
    """
    try:
        count = 0
        with open(input_path, encoding="utf-8") as infile, open(
            output_path, "w", encoding="utf-8"
        ) as outfile:
            logger.info(f"Reading from {input_path}...")
            for line in infile:
                if count < n_lines:
                    # Optional: Validate if the line is valid JSON
                    try:
                        json.loads(line)  # Try parsing to check validity
                        outfile.write(line)  # Write the original line
                        count += 1
                    except json.JSONDecodeError:
                        logger.info(
                            f"Warning: Skipping invalid JSON line: {line.strip()}"
                        )
                else:
                    break  # Stop after reading N lines
            logger.info(f"Successfully saved {count} lines to {output_path}")

    except FileNotFoundError:
        logger.info(f"Error: Input file not found at {input_path}")
    except Exception as e:
        logger.info(f"An error occurred: {e}")


def apply_shuffle_sql(data):
    """Shuffles the words in the SQL query."""
    modified_data = copy.deepcopy(data)  # Work on a copy
    sql_query = modified_data.get("SQL", "")
    if sql_query:
        # Simple split by space, might need more sophisticated tokenization for complex SQL
        tokens = sql_query.split()
        random.shuffle(tokens)
        modified_data["SQL"] = " ".join(tokens)
    return modified_data


def apply_question_crop_middle(data, crop_ratio=0.3):
    """Removes a middle portion of the question text."""
    modified_data = copy.deepcopy(data)  # Work on a copy
    question = modified_data.get("question", "")
    if question:
        words = question.split()
        n_words = len(words)
        if n_words > 3:  # Only crop if there are enough words
            # Calculate start and end index for removal (middle portion)
            remove_count = int(n_words * crop_ratio)
            if (
                remove_count == 0 and n_words > 3
            ):  # Ensure at least one word is removed if possible
                remove_count = 1

            start_remove_index = (n_words - remove_count) // 2
            end_remove_index = start_remove_index + remove_count

            # Keep words before and after the removed section
            kept_words = words[:start_remove_index] + words[end_remove_index:]
            modified_data["question"] = " ".join(kept_words)
        # Else: keep short questions as they are
    return modified_data


def save_augmented_train_sets(augmentation_functions, input_file_path):
    output_dir = os.path.dirname(input_file_path)  # Use the same directory as input

    # Get base filename and extension
    base_name, ext = os.path.splitext(
        os.path.basename(input_file_path)
    )  # e.g., ('train', '.jsonl')

    # Prepare output file handles
    output_files = {}
    output_paths = {}

    saved_files = []

    try:
        # Create output file handles for each manipulation
        for aug_name in augmentation_functions.keys():
            output_filename = f"{base_name}_{aug_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            output_paths[aug_name] = output_path
            # Open in 'w' (write mode) to clear the file if it exists,
            # or use 'a' (append mode) if you might run this multiple times partially
            output_files[aug_name] = open(output_path, "w", encoding="utf-8")
            logger.info(f"Output for '{aug_name}' will be saved to: {output_path}")

        # Read input and process line by line
        logger.info(f"\nProcessing input file: {input_file_path}")
        with open(input_file_path, encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                try:
                    original_data = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.info(f"Warning: Skipping invalid JSON on line {i + 1}")
                    continue

                # Apply each manipulation to the original data for this line
                for aug_name in augmentation_functions.keys():
                    if (
                        aug_name in output_files
                    ):  # Check if we have a file handle for it
                        manip_func = augmentation_functions[aug_name]
                        # Apply the manipulation - function already makes a deep copy
                        modified_data = manip_func(original_data)

                        # Write the modified data to the corresponding output file
                        json.dump(modified_data, output_files[aug_name])
                        output_files[aug_name].write(
                            "\n"
                        )  # Add newline for JSONL format
                        saved_files.append(output_paths[aug_name])

        logger.info("\nProcessing finished.")

    except FileNotFoundError:
        logger.info(f"Error: Input file not found at {input_file_path}")
        exit(0)

    finally:
        # Ensure all output files are closed
        for f_handle in output_files.values():
            if f_handle and not f_handle.closed:
                f_handle.close()
        logger.info("Output files closed.")

    return saved_files


def convert_to_jsonl_and_save_short_version(n_lines_short, base_files):
    for infile_path in base_files.values():
        infile_content = json.load(open(infile_path))
        outfile_path = infile_path.replace(".json", ".jsonl")

        with open(outfile_path, "w", encoding="utf-8") as outfile:
            for i, entry in enumerate(infile_content):
                # Convert dictionary to JSON string
                # ensure_ascii=False allows non-ASCII characters directly
                if "question_id" not in entry.keys():
                    entry["question_id"] = i
                entry["split"] = outfile_path.split("/")[-1].split(".")[0]
                json_line = json.dumps(entry, ensure_ascii=False)
                # Write the JSON string followed by a newline character
                outfile.write(json_line + "\n")

        save_first_n_lines_jsonl(
            input_path=outfile_path,
            output_path=outfile_path.replace(".jsonl", "_short.jsonl"),
            n_lines=n_lines_short,
        )
