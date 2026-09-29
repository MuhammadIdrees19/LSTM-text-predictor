import os

# Set before TensorFlow is imported. Importing this package loads the model.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from utils.pdf_extractor import extract_text_from_pdf
from utils.text_cleaner import (
    clean_text,
    tokenize_text,
    pad_input_sequences,
    add_short_context_copies,
)
from utils.vocabulary import create_vocabulary
from src.model import build_model
from src.trainer import (
    compile_and_train,
    evaluate_topk_accuracy,
    predict_next_word,
    predict_next_words,
)
from utils.analyzer import analyze_vocabulary
