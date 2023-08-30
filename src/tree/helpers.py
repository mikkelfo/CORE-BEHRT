import os
import pandas as pd
import src.common.loading as loading
from src.tree.node import Node


def get_counts(cfg):
    train, val = loading.encoded(cfg)
    vocabulary = loading.vocabulary(cfg)

    inv_vocab = {v: k for k, v in vocabulary.items()}
    counts = {}

    for codes in train["concept"] + val["concept"]:
        for code in codes:
            counts[inv_vocab[code]] = counts.get(inv_vocab[code], 0) + 1

    return counts


def build_tree(cfg, counts=None):
    files = [os.path.join(cfg.paths.dump_dir, file) for file in cfg.paths.dump_files]
    codes = create_levels(files)
    tree = create_tree(codes)

    if cfg.tree.max_level is not None:
        tree.cutoff_at_level(cfg.tree.max_level)
        tree.extend_leaves(cfg.tree.max_level)

    if counts is None:
        counts = loading.load(cfg.paths.extra_dir, cfg.paths.base_counts)

    tree.base_counts(counts)
    tree.sum_counts()
    tree.redist_counts()
    return tree


def create_levels(
    files: list = [
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
        (2, "BG_GENDER_Mand"),
        (2, "BG_GENDER_Kvinde"),
        (1, "[BMI]"),
        (2, "BG_BMI_underweight"),
        (2, "BG_BMI_normal"),
        (2, "BG_BMI_overweight"),
        (2, "BG_BMI_obese"),
        (2, "BG_BMI_extremely-obese"),
        (2, "BG_BMI_morbidly-obese"),
        (2, "BG_BMI_nan"),
    ]
    codes.extend(background)

    return codes


def create_tree(codes: list):
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
