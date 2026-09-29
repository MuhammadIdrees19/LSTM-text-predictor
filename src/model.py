from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout


def build_model(
    vocab_size,
    context_size=10,
    embedding_dim=128,
    lstm_units=128,
    lstm_dropout=0.1,
    dense_dropout=0.2,
):
    """
    Build the next-word prediction model.

    Architecture:

    Integer IDs
        ↓
    Embedding (with mask_zero=True for padding)
        ↓
    LSTM
        ↓
    Dropout (regularization to prevent overfitting)
        ↓
    Dense + Softmax

    vocab_size here should be the TOTAL vocabulary size including
    PAD and UNK tokens. The Embedding and Dense layers use vocab_size + 1
    so every id has a row even if a padding index is unused.

    Dropout is intentionally light. On a few thousand tokens, LSTM dropout
    of 0.3 plus recurrent dropout stopped training while the network was
    still worse than a bigram table: the previous word never reliably
    reached the softmax. Recurrent dropout is 0 so the memory cell can
    carry the last word forward. The Dense dropout still limits memorization
    of the full 10-word window.
    """

    model = Sequential([
        Input(shape=(context_size,), dtype="int32"),

        Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            mask_zero=True
        ),

        LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=0.0),

        Dropout(dense_dropout),

        Dense(
            vocab_size + 1,
            activation="softmax"
        )
    ])

    return model
