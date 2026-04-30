import hashlib
import random
import re


LABELS = ["A", "B", "C", "D"]


def afri_mcqa_process_docs(dataset):
    audio_columns = [col for col in dataset.column_names if "audio" in col]
    if audio_columns:
        dataset = dataset.remove_columns(audio_columns)

    keep_indices = [idx for idx, image in enumerate(dataset["image"]) if image is not None]
    if len(keep_indices) != len(dataset):
        dataset = dataset.select(keep_indices)

    return dataset


def _get_shuffled_choices(doc):
    choices = [
        doc["correct_native"],
        doc["wrong_native_o1"],
        doc["wrong_native_o2"],
        doc["wrong_native_o3"],
    ]
    seed = int(hashlib.md5(doc["ID"].encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices


def _get_labeled_choices(doc):
    return list(zip(LABELS, _get_shuffled_choices(doc)))


# def _extract_answer_letter(text):
#     text = text.strip().upper()
#     candidates = []

#     for char in [",", ".", "!", "?", ";", ":", "'", '"']:
#         text = text.strip(char)
#     padded = f" {text} "

#     for label in LABELS:
#         if f"({label})" in padded:
#             candidates.append((padded.rfind(f"({label})"), label))

#     for label in LABELS:
#         if re.search(rf"(?:^|\s){label}(?:\s|$)", padded):
#             candidates.append((padded.rfind(f" {label} "), label))

#     for label in LABELS:
#         dot_match = re.search(rf"(?:^|\s){label}[\.)：:]", padded)
#         if dot_match:
#             candidates.append((dot_match.start(), label))

#     answer_match = re.search(r"ANSWER\s*[：:]\s*([A-D])", padded)
#     if answer_match:
#         candidates.append((answer_match.start(), answer_match.group(1)))

#     if not candidates:
#         return ""

#     return max(candidates, key=lambda item: item[0])[1]


def _extract_answer_letter(text):
    LABELS = ["A", "B", "C", "D"]
    text = text.strip().upper()
    candidates = []

    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(char)
    padded = f" {text} "

    for label in LABELS:
        if f"({label})" in padded:
            candidates.append((padded.find(f"({label})"), label))

    for label in LABELS:
        match = re.search(rf"(?:^|\s){label}(?:\s|$)", padded)
        if match:
            candidates.append((match.start(), label))

    for label in LABELS:
        dot_match = re.search(rf"(?:^|\s){label}[\.)：:]", padded)
        if dot_match:
            candidates.append((dot_match.start(), label))

    answer_match = re.search(r"ANSWER\s*[：:]\s*([A-D])", padded)
    if answer_match:
        candidates.append((answer_match.start(), answer_match.group(1)))

    if not candidates:
        return ""

    return min(candidates, key=lambda item: item[0])[1]

def afri_mcqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def afri_mcqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    labeled_choices = _get_labeled_choices(doc)
    option_map = {label: choice for label, choice in labeled_choices}

    country = doc["Country"].strip()
    question = doc["native_question"].strip()

    # Old prompt logic kept here for fallback comparison.
    # instruction = "Select the correct answer from the options below. Respond with only the option letter (A, B, C, or D)."
    # return f"{question}\n" + "\n".join([f"{label}. {choice}" for label, choice in labeled_choices]) + f"\n{instruction}\nAnswer:"

    return (
        "You are given an image. Analyze the image and answer the following multiple-choice question. "
        "Only one option is correct. Return only the correct option name i.e. A, B, C or D. "
        f"The question is relevant to {country}.\n"
        f"Question: {question}\n"
        f"Options: A. {option_map['A']}, B. {option_map['B']}, C. {option_map['C']}, D. {option_map['D']}\nAnswer:"
    )


def afri_mcqa_doc_to_target(doc):
    for label, choice in _get_labeled_choices(doc):
        if choice == doc["correct_native"]:
            return label
    raise ValueError("Correct answer not found in shuffled choices")


def afri_mcqa_process_results(doc, results):
    pred = _extract_answer_letter(results[0])
    gold = afri_mcqa_doc_to_target(doc)
    return {"afri_mcqa_acc": 1.0 if pred == gold else 0.0}

def _get_shuffled_choices_en(doc):
    choices = [
        doc["correct_en"],
        doc["wrong_en_o1"],
        doc["wrong_en_o2"],
        doc["wrong_en_o3"],
    ]
    seed = int(hashlib.md5(doc["ID"].encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices


def _get_labeled_choices_en(doc):
    return list(zip(LABELS, _get_shuffled_choices_en(doc)))


def afri_mcqa_en_doc_to_text(doc, model_specific_prompt_kwargs=None):
    labeled_choices = _get_labeled_choices_en(doc)
    option_map = {label: choice for label, choice in labeled_choices}

    country = doc["Country"].strip()
    question = doc["eng_question"].strip()

    return (
        "You are given an image. Analyze the image and answer the following multiple-choice question. "
        "Only one option is correct. Return only the correct option name i.e. A, B, C or D. "
        f"The question is relevant to {country}.\n"
        f"Question: {question}\n"
        f"Options: A. {option_map['A']}, B. {option_map['B']}, C. {option_map['C']}, D. {option_map['D']}\nAnswer:"
    )


def afri_mcqa_en_doc_to_target(doc):
    for label, choice in _get_labeled_choices_en(doc):
        if choice == doc["correct_en"]:
            return label
    raise ValueError("Correct English answer not found in shuffled choices")


def afri_mcqa_en_process_results(doc, results):
    pred = _extract_answer_letter(results[0])
    gold = afri_mcqa_en_doc_to_target(doc)
    return {"afri_mcqa_acc": 1.0 if pred == gold else 0.0}

