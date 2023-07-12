import pandas as pd
import torch
from src.tree.node import Node


# TODO: Needs config file?
def get_counts():
    train = torch.load("data/processed/train_encoded.pt")
    val = torch.load("data/processed/val_encoded.pt")
    vocabulary = torch.load("data/processed/vocabulary.pt")
    inv_vocab = {v: k for k, v in vocabulary.items()}

    counts = dict()

    for codes in train["concept"]:
        for code in codes:
            counts[inv_vocab[code]] = counts.get(inv_vocab[code], 0) + 1
    for codes in val["concept"]:
        for code in codes:
            counts[inv_vocab[code]] = counts.get(inv_vocab[code], 0) + 1

    return counts


def build_tree(
    files=[
        "data/data_dumps/sks_dump_diagnose.xlsx",
        "data/data_dumps/sks_dump_medication.xlsx",
    ],
    counts=None,
    cutoff_level=5,
    extend_level=5,
):
    codes = create_levels(files)
    tree = create_tree(codes)
    if cutoff_level is not None:
        tree.cutoff_at_level(cutoff_level)
    if extend_level is not None:
        tree.extend_leaves(extend_level)

    if counts is None:
        counts = torch.load("data/extra/base_counts.pt")
    tree.base_counts(counts)
    tree.sum_counts()
    tree.redist_counts()
    return tree


def create_levels(
    files=[
        "data/data_dumps/sks_dump_diagnose.xlsx",
        "data/data_dumps/sks_dump_medication.xlsx",
    ]
):
    codes = []
    for file in files:
        df = pd.read_excel(file)

        level = -1
        prev_code = ""
        for i, (code, text) in df.iterrows():
            if pd.isna(code):  # Only for diagnosis
                # Manually set nan codes for Chapter and Topic (as they have ranges)
                if text[:3].lower() == "kap":
                    code = "XX"  # Sets Chapter as level 2 (XX)
                else:
                    if pd.isna(
                        df.iloc[i + 1].Kode
                    ):  # Skip "subsub"-topics (double nans not started by chapter)
                        continue
                    code = "XXX"  # Sets Topic as level 3 (XXX)

            level += len(code) - len(
                prev_code
            )  # Add distance between current and previous code to level
            prev_code = code  # Set current code as previous code

            if code.startswith("XX"):  # Gets proper code (chapter/topic range)
                code = text.split()[-1]

            # Needed to fix the levels for medication
            if "medication" in file and level in [3, 4, 5]:
                codes.append((level - 1, code))
            elif "medication" in file and level == 7:
                codes.append((level - 2, code))
            else:
                codes.append((level, code))

    # Add background
    background = [
        (0, "BG"),
        (1, "[GENDER]"),
        (2, "BG_Mand"),
        (2, "BG_Kvinde"),
        (1, "[BMI]"),
        (2, "BG_underweight"),
        (2, "BG_normal"),
        (2, "BG_overweight"),
        (2, "BG_obese"),
        (2, "BG_extremely-obese"),
        (2, "BG_morbidly-obese"),
        (2, "BG_nan"),
    ]
    codes.extend(background)

    return codes


def create_tree(codes):
    root = Node("root")
    parent = root
    for i in range(len(codes)):
        level, code = codes[i]
        next_level = codes[i + 1][0] if i < len(codes) - 1 else level
        dist = next_level - level

        if dist >= 1:
            for _ in range(dist):
                parent.add_child(code)
                parent = parent.children[-1]
        elif dist <= 0:
            parent.add_child(code)
            for _ in range(0, dist, -1):
                parent = parent.parent
    return root
