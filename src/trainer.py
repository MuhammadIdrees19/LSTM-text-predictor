import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.text_cleaner import clean_text, pad_input_sequences, tokenize_text


def compile_and_train(model, X_train, y_train, X_val, y_val,
                      epochs=40, batch_size=32, learning_rate=0.002):
    """
    Compile and train the model with appropriate callbacks.

    The learning rate is higher than a cautious 0.0005 because heavy
    regularization was ending training before local word transitions
    (the signal a bigram model uses) showed up in the weights.

    Early stopping watches validation accuracy, not validation loss.
    On this document the validation slice is a later section, so loss
    bottoms out while the model still mostly predicts "the" and "a".
    Accuracy keeps rising for a few more epochs as previous-word
    patterns transfer, and those are the weights we keep.

    Callbacks:
    - EarlyStopping: stops when val_accuracy stops improving
    - ReduceLROnPlateau: reduces learning rate when val_accuracy plateaus
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_topk_accuracy(model, X_test, y_test, k_values=(3, 5)):
    """
    Evaluate top-k accuracy on test data.

    Top-k accuracy means the correct word is among the top k predictions.
    """
    predictions = model.predict(X_test, verbose=0)
    results = {}

    for k in k_values:
        top_k_preds = tf.math.top_k(predictions, k=k).indices  # shape: [n, k]
        # Reshape y_test to [n, 1] for broadcasting comparison
        y_expanded = tf.cast(tf.reshape(y_test, [-1, 1]), tf.int32)
        # Check if correct label is in any of the top-k positions
        correct = tf.reduce_any(
            tf.equal(y_expanded, tf.cast(top_k_preds, tf.int32)),
            axis=1
        )
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        results[f"top_{k}_accuracy"] = float(accuracy.numpy())

    return results


def predict_next_word(text, model, word_to_id, context_size=10, top_k=5):
    """
    Predict the next word for a given text.

    Uses the same cleaning and tokenization as training, then keeps the
    last `context_size` tokens. <PAD> and <UNK> are not returned: they are
    not words a user asked to predict.

    Returns:
    - predicted_word: the top-1 predicted word
    - top_predictions: list of (word, probability) for top-k predictions
    """
    cleaned_input = clean_text(text)
    input_tokens = tokenize_text(cleaned_input)
    input_tokens = input_tokens[-context_size:]

    id_to_word = {word_id: word for word, word_id in word_to_id.items()}

    unknown_id = word_to_id["<UNK>"]
    pad_id = word_to_id["<PAD>"]
    input_ids = [word_to_id.get(word, unknown_id) for word in input_tokens]

    padded_input = pad_input_sequences([input_ids], context_size=context_size)

    predictions = model.predict(padded_input, verbose=0)[0]

    # Ask for a few extra ranks so filtering PAD/UNK still leaves top_k words.
    ranked_ids = tf.argsort(predictions, direction="DESCENDING").numpy()

    top_predictions = []
    for word_id in ranked_ids:
        word_id = int(word_id)
        if word_id in (pad_id, unknown_id):
            continue
        word = id_to_word.get(word_id)
        if word is None:
            continue
        top_predictions.append((word, float(predictions[word_id])))
        if len(top_predictions) >= top_k:
            break

    predicted_word = top_predictions[0][0] if top_predictions else "<UNK>"
    return predicted_word, top_predictions


def predict_next_words(text, model, word_to_id, context_size=10, num_words=8, top_k=1):
    """
    Predict the next `num_words` words, one at a time.

    After each prediction the word is appended to the sentence and the model
    is called again. Only the last `context_size` tokens are used, matching
    training. The same cleaning rules apply at every step.
    """
    if num_words < 1:
        return []

    generated_words = []
    current_text = text.strip()

    for _ in range(num_words):
        next_word, _ = predict_next_word(
            current_text,
            model,
            word_to_id,
            context_size=context_size,
            top_k=top_k,
        )
        if not next_word or next_word == "<UNK>":
            break
        generated_words.append(next_word)
        current_text = f"{current_text} {next_word}"

    return generated_words
