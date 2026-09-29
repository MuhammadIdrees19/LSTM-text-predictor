import os

# Set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.pdf_extractor import extract_text_from_pdf as _extract_text_from_pdf
from utils.vocabulary import MIN_WORD_FREQUENCY, create_vocabulary
from utils.text_cleaner import (
    add_short_context_copies,
    clean_text,
    create_input_target_sequences,
    pad_input_sequences,
    tokenize_text,
)

from src.model import build_model

from src.trainer import compile_and_train, evaluate_topk_accuracy, predict_next_word, predict_next_words


# Fixed seed so a rerun follows the same training path.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------
# Paths
# --------------------------------
# Anchored to this file so the PDF is found even when the shell's
# current directory is not the project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "input" / "document.pdf"
EXTRACTED_TEXT_PATH = PROJECT_ROOT / "data" / "output" / "extracted_text.txt"
CLEANED_TEXT_PATH = PROJECT_ROOT / "data" / "output" / "cleaned_text.txt"


# Context window: number of previous words used to predict the next word
CONTEXT_SIZE = 10
EMBEDDING_DIM = 128
LSTM_UNITS = 128
LEARNING_RATE = 0.002
BATCH_SIZE = 32
EPOCHS = 40
NEXT_WORDS = 8


# --------------------------------
# Step 1: Extract text from PDF
# --------------------------------


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    return _extract_text_from_pdf(pdf_path)


# --------------------------------
# Step 2: Clean text
# --------------------------------


def clean_pipeline_text(text):
    """Clean extracted text for training."""
    return clean_text(text)


# --------------------------------
# Step 3: Word Tokenization
# --------------------------------


def tokenize_pipeline_text(cleaned_text):
    """Split cleaned text into word tokens."""
    return tokenize_text(cleaned_text)


# --------------------------------
# Step 4: Create Vocabulary
# --------------------------------


def build_vocabulary(tokens, min_frequency=MIN_WORD_FREQUENCY):
    """Build vocabulary from tokens with minimum frequency filtering."""
    word_counts, word_to_id = create_vocabulary(tokens, min_frequency=min_frequency)
    vocab_size = len(word_to_id)
    print(f"Vocabulary size: {vocab_size}")
    print(f"(Includes <PAD>=0 and <UNK>=1)")
    return word_counts, word_to_id


# --------------------------------
# Step 5: Create Input-Target Pairs
# --------------------------------


def create_sequences(tokens, word_to_id, context_size=CONTEXT_SIZE):
    """Create input-target pairs using a sliding context window."""
    input_sequences, target_ids = create_input_target_sequences(
        tokens, word_to_id, context_size=context_size
    )

    print(f"Total training pairs: {len(input_sequences)}")
    print(f"Skipped pairs where target was <UNK>.\n")

    return input_sequences, target_ids


# --------------------------------
# Step 6: Padding
# --------------------------------


def pad_sequences_pipeline(input_sequences, context_size=CONTEXT_SIZE):
    """Pad all input sequences to exactly context_size."""
    padded_inputs = pad_input_sequences(input_sequences, context_size=context_size)

    print(f"Padded input shape: {padded_inputs.shape}")

    return padded_inputs


# --------------------------------
# Step 7: Train / Validation / Test Split
# --------------------------------


def split_data(padded_inputs, target_ids):
    """Split data into train / validation / test sets (sequential, no shuffling)."""
    total_samples = len(padded_inputs)

    # 80% train, 10% validation, 10% test (sequential, no shuffling)
    train_end = int(total_samples * 0.80)
    val_end = int(total_samples * 0.90)

    X_train = padded_inputs[:train_end]
    y_train = target_ids[:train_end]

    X_val = padded_inputs[train_end:val_end]
    y_val = target_ids[train_end:val_end]

    X_test = padded_inputs[val_end:]
    y_test = target_ids[val_end:]

    # Short copies are training-only. Validation and test stay full 10-word windows,
    # so the reported accuracy is still "next word given 10 words of context".
    X_train, y_train = add_short_context_copies(
        X_train, y_train, context_size=CONTEXT_SIZE
    )

    print(f"Total samples:      {total_samples}")
    print(f"Training samples:   {len(X_train)} (includes short-context copies)")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------
# Step 8-10: Convert to tensors, Build Model, Train
# --------------------------------


def convert_to_tensors(X_train, y_train, X_val, y_val, X_test, y_test):
    """Convert numpy arrays to TensorFlow tensors."""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.int32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    X_val = tf.convert_to_tensor(X_val, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

    X_test = tf.convert_to_tensor(X_test, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model_pipeline(vocab_size, context_size=CONTEXT_SIZE, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS):
    """Build the next-word prediction model."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = build_model(
        vocab_size=vocab_size,
        context_size=context_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
    )
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """Compile and train the model."""
    history = compile_and_train(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    val_accuracies = history.history["val_accuracy"]
    best_epoch = max(range(len(val_accuracies)), key=val_accuracies.__getitem__)
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"  train accuracy: {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  val accuracy:   {val_accuracies[best_epoch]:.4f}")
    print(f"  val loss:       {history.history['val_loss'][best_epoch]:.4f}")

    return history


# --------------------------------
# Step 11: Evaluate on Test Data
# --------------------------------


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    topk_results = evaluate_topk_accuracy(model, X_test, y_test, k_values=(3, 5))

    print(f"Top-3 Accuracy: {topk_results['top_3_accuracy']:.4f} ({topk_results['top_3_accuracy'] * 100:.2f}%)")
    print(f"Top-5 Accuracy: {topk_results['top_5_accuracy']:.4f} ({topk_results['top_5_accuracy'] * 100:.2f}%)")

    return test_loss, test_accuracy, topk_results


# --------------------------------
# Step 12: Prediction Tests
# --------------------------------


def run_prediction_tests(model, word_to_id, context_size=CONTEXT_SIZE, top_k=5, next_words=8):
    """Run prediction tests with sample prompts."""
    test_prompts = [
        "Tokenization helps machines understand and process human language by breaking it",
        "LSTM is a special type",
        "machine learning is a subfield of",
        "the model learns from context",
        "word embedding is a technique that converts",
    ]

    for prompt in test_prompts:
        predicted_word, top_predictions = predict_next_word(
            prompt, model, word_to_id, context_size=context_size, top_k=top_k
        )
        continuation = predict_next_words(
            prompt,
            model,
            word_to_id,
            context_size=context_size,
            num_words=next_words,
        )

        print(f"\nInput: {prompt}")
        print(f"Predicted next word: {predicted_word}")
        print(f"Next {next_words} words: {' '.join(continuation)}")
        print("Top-5 predictions:")
        for rank, (word, prob) in enumerate(top_predictions, 1):
            print(f"  {rank}. {word:20} ({prob:.4f})")


# --------------------------------
# Step 13: User sentence prediction
# --------------------------------


def run_user_prediction(model, word_to_id, sentence, context_size=CONTEXT_SIZE, next_words=NEXT_WORDS):
    """Predict the next words for one user sentence."""
    return predict_next_words(
        sentence,
        model,
        word_to_id,
        context_size=context_size,
        num_words=next_words,
    )