import os
import sys

# Set before TensorFlow is imported (src.engine imports it).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from utils.analyzer import analyze_vocabulary
from utils.vocabulary import MIN_WORD_FREQUENCY
from src.engine import (
    BATCH_SIZE,
    CLEANED_TEXT_PATH,
    CONTEXT_SIZE,
    EPOCHS,
    EXTRACTED_TEXT_PATH,
    LEARNING_RATE,
    NEXT_WORDS,
    PDF_PATH,
    build_model_pipeline,
    build_vocabulary,
    clean_pipeline_text,
    convert_to_tensors,
    create_sequences,
    evaluate_model,
    extract_text_from_pdf,
    pad_sequences_pipeline,
    run_prediction_tests,
    run_user_prediction,
    split_data,
    tokenize_pipeline_text,
    train_model,
)


def main():
    """Run extraction, training, evaluation, and prediction."""
    print("=" * 60)
    print("STEP 1: Extracting text from PDF")
    print("=" * 60)

    text = extract_text_from_pdf(PDF_PATH)
    EXTRACTED_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTED_TEXT_PATH.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text)} characters from PDF.\n")

    print("=" * 60)
    print("STEP 2: Cleaning text")
    print("=" * 60)

    cleaned_text = clean_pipeline_text(text)
    CLEANED_TEXT_PATH.write_text(cleaned_text, encoding="utf-8")
    print(f"Cleaned text: {len(cleaned_text)} characters.\n")

    print("=" * 60)
    print("STEP 3: Tokenizing text")
    print("=" * 60)

    tokens = tokenize_pipeline_text(cleaned_text)
    print(f"Total tokens: {len(tokens)}\n")

    print("=" * 60)
    print("STEP 4: Building vocabulary")
    print("=" * 60)

    word_counts, word_to_id = build_vocabulary(tokens, min_frequency=MIN_WORD_FREQUENCY)
    vocab_size = len(word_to_id)
    analyze_vocabulary(word_counts, word_to_id, min_frequency=MIN_WORD_FREQUENCY)
    print()

    print("Top 20 most frequent words:")
    for word, frequency in word_counts.most_common(20):
        word_id = word_to_id.get(word, "N/A")
        print(f"  {word:20} freq={frequency:4d}  id={word_id}")

    print("\nFirst 20 word IDs:")
    for word, word_id in list(word_to_id.items())[:20]:
        print(f"  {word:20} {word_id}")

    print("\n" + "=" * 60)
    print("STEP 5: Creating input-target sequences")
    print("=" * 60)

    input_sequences, target_ids = create_sequences(
        tokens, word_to_id, context_size=CONTEXT_SIZE
    )

    print("=" * 60)
    print("STEP 6: Padding sequences")
    print("=" * 60)

    padded_inputs = pad_sequences_pipeline(input_sequences, context_size=CONTEXT_SIZE)

    print("\n" + "=" * 60)
    print("STEP 7: Splitting data")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(padded_inputs, target_ids)

    print("\n" + "=" * 60)
    print("STEP 8: Converting to tensors")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = convert_to_tensors(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "=" * 60)
    print("STEP 9: Building model")
    print("=" * 60)

    model = build_model_pipeline(vocab_size=vocab_size)

    print("\n" + "=" * 60)
    print("STEP 10: Training model")
    print("=" * 60)

    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Max epochs:    {EPOCHS}")
    print("Early stopping restores the best validation-accuracy weights.\n")

    train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    print("\n" + "=" * 60)
    print("STEP 11: Evaluating on test data")
    print("=" * 60)

    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("STEP 12: Prediction tests")
    print("=" * 60)

    run_prediction_tests(
        model, word_to_id, context_size=CONTEXT_SIZE, top_k=5, next_words=NEXT_WORDS
    )

    print("\n" + "=" * 60)
    print("STEP 13: User sentence")
    print("=" * 60)

    if sys.stdin.isatty():
        print(f"Type a sentence to predict the next {NEXT_WORDS} words.")
        print("Press Enter on a blank line to stop.")

        while True:
            try:
                sentence = input("\nInput: ").strip()
            except EOFError:
                break
            if not sentence:
                break

            continuation = run_user_prediction(
                model,
                word_to_id,
                sentence,
                context_size=CONTEXT_SIZE,
                next_words=NEXT_WORDS,
            )
            print(f"Output: {' '.join(continuation)}")
    else:
        print("Skipped: stdin is not a terminal.")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
