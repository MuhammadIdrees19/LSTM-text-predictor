import re


# Meaningful single-character words to keep in vocabulary
KEEP_SINGLE_CHARS = {"a", "i"}

# Leftover symbols from LSTM equations (ht, Ct, Wf, bf, ...). After punctuation
# is stripped they sit between real words and break patterns such as
# "forget gate decides", which the later GRU section also uses.
FORMULA_VARIABLES = {
    "ht", "ct", "ft", "xt", "ot",
    "wf", "bf", "wi", "bi", "wc", "bc", "wo", "bo",
}


def _is_header_or_page(line):
    lowered = line.lower()
    if "compiled study notes" in lowered:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    if re.fullmatch(r"[\s.·]+", line):
        return True
    return False


def _is_equation_line(line):
    """Drop lines that are formulas rather than sentences.

    Prose that merely contains an equals sign, such as
    "Scikit-learn = traditional Machine Learning", has enough real words
    to keep. A line like "ft = σ(Wf[ht−1, xt] + bf)" does not.
    """
    prose_words = re.findall(r"[A-Za-z]{3,}", line)
    has_math = ("=" in line) or any(symbol in line for symbol in "σ⊙⊗")
    return has_math and len(prose_words) < 5


def clean_text(text):
    """
    Clean extracted text for next-word prediction training.

    Steps:
    1. Remove page headers/footers and standalone page numbers
    2. Remove the table of contents (headings repeated again in the body)
    3. Remove formula lines that are not sentences
    4. Remove leading section numbers
    5. Drop short parenthetical abbreviations so they are not inserted
       into the sentence ("Machine Learning (ML) is" -> "machine learning is")
    6. Remove formatting symbols, Greek letters, and other non-language characters
    7. Normalize whitespace and lowercase
    """
    lines = text.split("\n")
    cleaned_lines = []
    in_table_of_contents = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if _is_header_or_page(line):
            continue

        # The contents block repeats every heading and then the body repeats
        # them again. Keeping both teaches the model heading-to-heading jumps.
        if line.lower() == "contents":
            in_table_of_contents = True
            continue

        if in_table_of_contents:
            # TOC entries are short headings or dot leaders. The body starts
            # at the first full prose line.
            if len(line.split()) < 12 or line.count(".") >= 3:
                continue
            in_table_of_contents = False

        if _is_equation_line(line):
            continue

        # "1 machine learning" / "2.1 scikit-learn" -> the heading words only
        line = re.sub(r"^\d+(\.\d+)*\s+", "", line)

        if not line.strip():
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = text.lower()

    for symbol in ("→", "—", "–", "−", "=", "•", "·"):
        text = text.replace(symbol, " ")

    # Parentheses are abbreviations or expansions inside the sentence, as in
    # "LSTM (Long Short-Term Memory) is a special type". Dropping them keeps
    # the words in the order a person types, so the continuation can follow.
    text = re.sub(r"\([^)]*\)", " ", text)

    text = re.sub(r"\[[^\]]*\]", " ", text)

    for character in (";", "/", "'", ",", ".", "!", "?", '"', ":"):
        text = text.replace(character, " ")

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text):
    """
    Split cleaned text into word tokens.

    Drops formula variables, pure numbers, and single-character leftovers
    from equations. Keeps "a" and "i".
    """
    tokens = []
    for token in text.split():
        if token in FORMULA_VARIABLES:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        if len(token) == 1 and token not in KEEP_SINGLE_CHARS:
            continue
        tokens.append(token)
    return tokens


def create_input_target_sequences(tokens, word_to_id, context_size=10):
    """
    Create input-target pairs using a sliding context window.

    For tokens shorter than context_size, the input is left-padded with 0.
    For tokens longer than context_size, only the latest context_size words are used.

    Targets mapped to <UNK> are skipped. Those words are too rare to learn,
    and training on them teaches the model to emit an unknown token.
    Unknown words may still appear inside the context window.
    """
    input_sequences = []
    target_ids = []
    unknown_id = word_to_id["<UNK>"]

    for i in range(1, len(tokens)):
        start = max(0, i - context_size)
        input_words = tokens[start:i]
        target_word = tokens[i]

        input_ids = [word_to_id.get(word, unknown_id) for word in input_words]
        target_id = word_to_id.get(target_word, unknown_id)

        if target_id == unknown_id:
            continue

        input_sequences.append(input_ids)
        target_ids.append(target_id)

    return input_sequences, target_ids


def add_short_context_copies(padded_inputs, target_ids, context_size=10, lengths=(1, 2, 3)):
    """
    Add training copies that keep only the last few words.

    A 10-word window from this document almost never repeats in a later
    section, so the network can memorize long prefixes that do not help
    the held-out text. Copies that show only the last 1, 2, or 3 words
    (with padding in front) practice the short transitions that do repeat,
    such as a gate name followed by "decides". The original full windows
    stay in the set, and this must be applied to training data only.
    """
    import numpy as np

    padded_inputs = np.asarray(padded_inputs)
    target_ids = np.asarray(target_ids)

    extra_inputs = []
    extra_targets = []
    for row, target in zip(padded_inputs, target_ids):
        for length in lengths:
            if length >= context_size:
                continue
            short = np.zeros_like(row)
            short[-length:] = row[-length:]
            extra_inputs.append(short)
            extra_targets.append(target)

    if not extra_inputs:
        return padded_inputs, target_ids

    combined_inputs = np.concatenate([padded_inputs, np.stack(extra_inputs)])
    combined_targets = np.concatenate([
        target_ids,
        np.asarray(extra_targets, dtype=target_ids.dtype),
    ])
    return combined_inputs, combined_targets


def pad_input_sequences(input_sequences, context_size=10):
    """
    Pad all input sequences to exactly `context_size`.

    Padding is added to the beginning (left) of the sequence.
    PAD ID = 0
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    padded_sequences = pad_sequences(
        input_sequences,
        maxlen=context_size,
        padding="pre",
        truncating="pre",
        value=0
    )

    return padded_sequences