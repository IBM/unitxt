import itertools
import re
from typing import Dict

from .generator_utils import ReusableGenerator
from .logging_utils import get_logger
from .random_utils import new_random_generator
from .stream import MissingStreamError, Stream

logger = get_logger()


def parse_random_mix_string(input_str):
    """Parses a string of format "source1[percentage1%]+source2[value2]+..." and returns a dictionary.

    Args:
        input_str (str): A string containing source names and their respective proportions. The format is
                         "source[proportion%]" or "source[proportion]", with multiple sources separated by "+".
                         The proportion can be a percentage (e.g., "90%") or a decimal number (e.g., "0.7").
                         If the proportion is not provided, it assumes 100%.

    Returns:
        dict: A dictionary where the keys are the source names and the values are the proportions converted to floats.
              If the proportion was given as a percentage, the value is divided by 100.

    Raises:
        ValueError: If the input string is not in the correct format.

    Example:
        >>> parse_random_mix_string("dale[90%]+oren[0.7]+mike")
            {'dale': 0.9, 'oren': 0.7, 'mike': 1.0}
    """
    if not re.fullmatch(
        r"((\w+\[\d*\.?\d*%?\]|\w+)\+)*(\w+\[\d*\.?\d*%?\]|\w+)",
        input_str,
    ):
        raise ValueError(f"Invalid input format for split '{input_str}'")

    pattern = re.compile(r"(\w+)(\[\d*\.?\d*%?\])?")
    matches = pattern.findall(input_str)

    return {
        name: float(value.strip("[]%")) / 100
        if "%" in value
        else (float(value.strip("[]")) if value else 1.0)
        for name, value in matches
    }


def parse_slices_string(input_str):
    """Parses a string of format "source1[value1:value2] + source2[value2:] + source3 + ..." and returns a dictionary.

    {"source1": [(value1,value2)], "source2": [(value2, None)], "source3": [(None,None)]...}.

    If a source appears multiple times with different indices, all index pairs are included in the list.

    Args:
        input_str (str): A string containing source names and their respective indices. The format is
                         "source[:index]" or "source[index:]", with multiple sources separated by "+".
                         The index represents the items to be taken from the source.

    Returns:
        dict: A dictionary where the keys are the source names and the values are lists of indices as tuples.
              If the index is before the colon, it is represented as (None, index),
              if it's after the colon, it's represented as (index, None)

    Raises:
        ValueError: If the input string is not in the correct format.

    Example:
        >>> parse_slices_string("oren[:50]+jake[24:]+test+oren[5:10]")
        {'oren': [(None, 50), (5, 10)], 'jake': [(24, None)], 'test': [(None, None)]}
    """
    result_dict = {}

    # Split the input string into a list of sources
    sources = re.split(r"\+", input_str)
    for source in sources:
        # If the source has a slice, parse it
        match = re.fullmatch(r"(\w+)\[(\d*):(\d*)\]", source)
        if match:
            name, start, end = match.groups()
            start = int(start) if start else None
            end = int(end) if end else None
        elif re.fullmatch(r"\w+", source):
            # If the source has no slice, use None for both start and end
            name = source
            start = end = None
        else:
            raise ValueError(
                f'The input string "{input_str}" is not in the correct format.'
            )

        if name not in result_dict:
            result_dict[name] = [(start, end)]
        else:
            result_dict[name].append((start, end))

    return result_dict


def slice_stream(stream, start, end):
    # If start is None, consume from the beginning
    if start is not None:
        stream = itertools.islice(stream, start, None)
    # If end is not None, consume until end
    if end is not None:
        stream = itertools.islice(stream, end)

    yield from stream
    # return stream


def slice_streams(input_streams, mapping):
    """Slices multiple input streams according to a mapping and chains the results together.

    Args:
        input_streams (dict): A dictionary where the keys are the names of the input streams
                              and the values are the input streams themselves.
        mapping (dict): A dictionary where the keys are the names of the new streams
                        and the values are dictionaries mapping old stream names
                        to lists of tuples representing slices.

    Returns:
        dict: A dictionary where the keys are the names of the new streams and the values are
              the new streams, which consist of parts of the old streams chained together.

    Raises:
        ValueError: If a stream is supposed to be sliced at an index greater than its length or a negative one.

    Example:
        >>> old_streams = {"train": [1, 2, 3, 4, 5, 6, 7, 8, 9], "test": [10, 11, 12, 13, 14]}
        >>> mapping = {"new_train": {"train": [(None, 5), (7, 9)]}, "new_test": {"test": [(2, None)]}}
        >>> slice_streams(old_streams, mapping)
        {"new_train": [1, 2, 3, 4, 5, 8, 9], "new_test": [12, 13, 14]}
    """
    new_streams = {}
    for new_stream, sources in mapping.items():

        def generator(new_stream, sources):
            for old_stream, slices in sources.items():
                if old_stream not in input_streams:
                    raise MissingStreamError(
                        f"'{old_stream}' is not available in input streams, but need to slice there from"
                    )
                old_stream_content = input_streams[old_stream]
                for start, end in slices:
                    yield from slice_stream(old_stream_content, start, end)

        new_streams[new_stream] = ReusableGenerator(
            generator, gen_kwargs={"new_stream": new_stream, "sources": sources}
        )

    return new_streams


def build_stream_routing(mapping):
    """Builds the stream mapping dictionary based on the provided mapping.

    The stream mapping dictionary represents the mapping of old streams to new streams
    and their respective probabilities. It ensures that the probabilities for each old stream
    do not sum up to more than one. If the sum of probabilities is less than one,
    a null stream (None) is included to account for the remaining probability.

    Args:
        mapping (dict): A dictionary specifying the mapping of old streams to new streams
                        and their respective probabilities.

    Returns:
        dict: A dictionary representing the stream mapping, where each entry corresponds to an
              old stream, and the value is a tuple containing the new streams and their respective
                probabilities.

    Example:
        >>> mapping = {
                'my_new_stream': {
                    'my_old_stream1': 0.6,
                    'my_old_stream2': 0.2
                },
                'my_new_stream2': {
                    'my_old_stream1': 0.4,
                    'my_old_stream2': 0.8
                }
            }
            stream_mapping = build_stream_mapping(mapping)
            logger.info(stream_mapping)
            # Output: {'my_old_stream1': (['my_new_stream', 'my_new_stream2'], [0.6, 0.4]),
            #          'my_old_stream2': (['my_new_stream', 'my_new_stream2'], [0.2, 0.8])}
    """
    stream_mapping = {}

    # Calculate total weight for each old stream
    total_weights = {}
    for _new_stream, old_streams in mapping.items():
        for old_stream, weight in old_streams.items():
            if old_stream not in total_weights:
                total_weights[old_stream] = weight
            else:
                total_weights[old_stream] += weight

    # Build stream_mapping with null stream included
    for new_stream, old_streams in mapping.items():
        for old_stream, weight in old_streams.items():
            if old_stream not in stream_mapping:
                stream_mapping[old_stream] = {}
            stream_mapping[old_stream][new_stream] = weight

            # Add null stream if total weight less than 1
            if total_weights[old_stream] < 1:
                stream_mapping[old_stream][None] = 1 - total_weights[old_stream]

    return {k: (list(v.keys()), list(v.values())) for k, v in stream_mapping.items()}


def rename_split(input_streams: Dict[str, Stream], mapping: Dict[str, str]):
    """Renames the streams.

    Args:
        input_streams (dict): A dictionary containing the input streams, where each key is
                              the name of the stream and the value is an iterable or generator
                              representing the stream.

        mapping (dict): A dictionary specifying the mapping of old streams to new streams.

    Returns:
        dict: A dictionary containing the generated new streams, where each key is the name
    of the new stream and the value is a generator representing the stream.
    """
    return {mapping.get(key, key): val for key, val in input_streams.items()}


def random_mix_generator(
    new_stream_name, new_stream_sources, stream_routing, input_streams
):
    for old_stream_name in new_stream_sources:
        optional_streams, weights = stream_routing[old_stream_name]
        random_generator = new_random_generator(sub_seed=old_stream_name)
        assert (
            old_stream_name in input_streams
        ), f"'{old_stream_name}' split not found.  Possibles options: {input_streams.keys()}"
        for item in input_streams[old_stream_name]:
            choice = random_generator.choices(optional_streams, weights=weights, k=1)[0]
            if choice == new_stream_name:
                yield item


def random_mix_streams(input_streams, mapping):
    """Creates new streams based on the provided input streams and mapping.

    The create_streams function generates new streams by selectively including items from
    the old streams based on the specified mapping. Each item will be included in at most
    one new stream, as defined by the probabilities in the mapping and stream routing.

    Args:
        input_streams (dict): A dictionary containing the input streams, where each key is
                              the name of the stream and the value is an iterable or generator
                              representing the stream.

        mapping (dict): A dictionary specifying the mapping of old streams to new streams
                        and their respective probabilities.

    Returns:
        dict: A dictionary containing the generated new streams, where each key is the name
              of the new stream and the value is a generator representing the stream.

    Example:
        >>> input_streams = {
                'my_old_stream1': gen1(),
                'my_old_stream2': gen2(),
            }
            mapping = {
                'my_new_stream': {
                    'my_old_stream1': 0.6,
                    'my_old_stream2': 0.2
                },
                'my_new_stream2': {
                    'my_old_stream1': 0.4,
                    'my_old_stream2': 0.8
                }
            }
            new_streams = create_streams(input_streams, mapping)
            for new_stream_name, new_stream in new_streams.items():
                logger.info(f"{new_stream_name}:")
                for _, item in zip(range(10), new_stream):
                    logger.info(item)
    """
    new_streams = {}

    # Build stream routing
    stream_routing = build_stream_routing(mapping)

    # Create new stream generators
    for new_stream_name, new_stream_sources in mapping.items():
        new_streams[new_stream_name] = ReusableGenerator(
            random_mix_generator,
            gen_kwargs={
                "new_stream_name": new_stream_name,
                "new_stream_sources": new_stream_sources,
                "stream_routing": stream_routing,
                "input_streams": input_streams,
            },
        )

    return new_streams


if __name__ == "__main__":
    logger.info(parse_random_mix_string("dale[90%]+oren[0.7]+mike"))
