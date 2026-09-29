def analyze_vocabulary(word_counts, word_to_id, min_frequency=1):
    """
    Print vocabulary coverage so it is obvious which words the model can learn.

    word_counts includes every token. word_to_id includes only words at or
    above min_frequency, plus <PAD> and <UNK>.
    """
    total_words = sum(word_counts.values())
    kept_occurrences = sum(
        count for word, count in word_counts.items() if count >= min_frequency
    )
    vocab_words = len(word_to_id) - 2  # exclude PAD and UNK
    filtered = [word for word, count in word_counts.items() if count < min_frequency]

    print("\nVocabulary Analysis:")
    print(f"  Total word occurrences: {total_words}")
    print(f"  Unique words in vocab:  {vocab_words}")
    print(f"  Coverage threshold:     min frequency >= {min_frequency}")
    if total_words:
        coverage = kept_occurrences / total_words
        print(f"  Token coverage:         {kept_occurrences}/{total_words} ({coverage:.1%})")
    print(f"  Filtered out:           {len(filtered)} words")

    if filtered:
        filtered_by_freq = sorted(filtered, key=lambda word: (-word_counts[word], word))
        preview = ", ".join(filtered_by_freq[:12])
        print(f"  Sample filtered words:  {preview}")

    return filtered