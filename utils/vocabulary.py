from collections import Counter


MIN_WORD_FREQUENCY = 1


def create_vocabulary(tokens, min_frequency=MIN_WORD_FREQUENCY):
    """
    Build vocabulary from tokens with minimum frequency filtering.

    ID 0 is reserved for <PAD> (padding).
    ID 1 is reserved for <UNK> (unknown words).

    Only words appearing at least `min_frequency` times are included.
    """
    word_counts = Counter(tokens)

    # Sort by frequency (most common first), then alphabetically for ties
    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    word_to_id = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    next_id = 2
    for word, frequency in sorted_words:
        if frequency >= min_frequency:
            word_to_id[word] = next_id
            next_id += 1

    return word_counts, word_to_id