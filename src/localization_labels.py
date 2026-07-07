"""
Coronary anatomical localization labels used by the SYNTAX annotations.

The dataset stores anatomical regions as numeric SYNTAX segment categories.
Class 0 is reserved for background in generated training masks.
"""

BACKGROUND_ID = 0
STENOSIS_CATEGORY_ID = 26
NUM_ANATOMY_CLASSES = 26  # background + 25 SYNTAX segment classes


SYNTAX_SEGMENTS = {
    1: {"code": "1", "name": "proximal RCA", "artery": "RCA", "group": "RCA proximal"},
    2: {"code": "2", "name": "mid RCA", "artery": "RCA", "group": "RCA mid"},
    3: {"code": "3", "name": "distal RCA", "artery": "RCA", "group": "RCA distal"},
    4: {"code": "4", "name": "posterior descending artery", "artery": "RCA/PDA", "group": "PDA"},
    5: {"code": "5", "name": "left main", "artery": "LM", "group": "LM"},
    6: {"code": "6", "name": "proximal LAD", "artery": "LAD", "group": "LAD proximal"},
    7: {"code": "7", "name": "mid LAD", "artery": "LAD", "group": "LAD mid"},
    8: {"code": "8", "name": "distal LAD", "artery": "LAD", "group": "LAD distal"},
    9: {"code": "9", "name": "first diagonal", "artery": "LAD", "group": "Diagonal"},
    10: {"code": "9a", "name": "first diagonal side branch", "artery": "LAD", "group": "Diagonal"},
    11: {"code": "10", "name": "second diagonal", "artery": "LAD", "group": "Diagonal"},
    12: {"code": "10a", "name": "second diagonal side branch", "artery": "LAD", "group": "Diagonal"},
    13: {"code": "11", "name": "proximal LCX", "artery": "LCX", "group": "LCX proximal"},
    14: {"code": "12", "name": "intermediate or anterolateral branch", "artery": "LCX", "group": "OM/intermediate"},
    15: {"code": "12a", "name": "obtuse marginal branch", "artery": "LCX", "group": "OM/intermediate"},
    16: {"code": "13", "name": "distal LCX", "artery": "LCX", "group": "LCX distal"},
    17: {"code": "14", "name": "left posterolateral branch", "artery": "LCX", "group": "PL branch"},
    18: {"code": "14a", "name": "left posterolateral side branch", "artery": "LCX", "group": "PL branch"},
    19: {"code": "15", "name": "left posterior descending artery", "artery": "LCX/PDA", "group": "PDA"},
    20: {"code": "16", "name": "right posterolateral branch", "artery": "RCA", "group": "PL branch"},
    21: {"code": "16a", "name": "right posterolateral side branch", "artery": "RCA", "group": "PL branch"},
    22: {"code": "16b", "name": "right posterolateral side branch", "artery": "RCA", "group": "PL branch"},
    23: {"code": "16c", "name": "right posterolateral side branch", "artery": "RCA", "group": "PL branch"},
    24: {"code": "12b", "name": "obtuse marginal side branch", "artery": "LCX", "group": "OM/intermediate"},
    25: {"code": "14b", "name": "left posterolateral side branch", "artery": "LCX", "group": "PL branch"},
}


def segment_label(segment_id: int) -> str:
    """Return a display label such as '6 proximal LAD'."""
    meta = SYNTAX_SEGMENTS.get(int(segment_id))
    if meta is None:
        return "unknown"
    return f"{meta['code']} {meta['name']}"


def segment_group(segment_id: int) -> str:
    """Return a grouped clinical location for real-time display."""
    meta = SYNTAX_SEGMENTS.get(int(segment_id))
    return "unknown" if meta is None else meta["group"]


def segment_artery(segment_id: int) -> str:
    """Return the major artery family for a segment id."""
    meta = SYNTAX_SEGMENTS.get(int(segment_id))
    return "unknown" if meta is None else meta["artery"]


ANATOMY_GROUPS = [
    "background",
    "RCA proximal",
    "RCA mid",
    "RCA distal",
    "PDA",
    "LM",
    "LAD proximal",
    "LAD mid",
    "LAD distal",
    "Diagonal",
    "LCX proximal",
    "OM/intermediate",
    "LCX distal",
    "PL branch",
]

ANATOMY_ARTERIES = [
    "background",
    "RCA",
    "RCA/PDA",
    "LM",
    "LAD",
    "LCX",
    "LCX/PDA",
]

GROUP_TO_ID = {name: idx for idx, name in enumerate(ANATOMY_GROUPS)}
ARTERY_TO_ID = {name: idx for idx, name in enumerate(ANATOMY_ARTERIES)}

SEGMENT_TO_GROUP_ID = [0] * NUM_ANATOMY_CLASSES
SEGMENT_TO_ARTERY_ID = [0] * NUM_ANATOMY_CLASSES
for _segment_id, _meta in SYNTAX_SEGMENTS.items():
    SEGMENT_TO_GROUP_ID[_segment_id] = GROUP_TO_ID[_meta["group"]]
    SEGMENT_TO_ARTERY_ID[_segment_id] = ARTERY_TO_ID[_meta["artery"]]


# ─── Merged (data-driven) label scheme ────────────────────────────────────────
#
# Rare/mutually-confused side-branch segments collapsed into the trunk segment
# they are consistently mistaken for, based on training-set support counts and
# the mask_localization confusion-matrix analysis (see
# confusion_mask_localization.py). 25 raw SYNTAX segments -> 14 merged classes.
#
# 9a, 10, 10a  -> 9   (Diagonal cluster; 9 has 198 train images vs 70/21/1)
# 16a,16b,16c  -> 16  (Right PL cluster; 16 has 231 train images vs 43/48/31)
# 12a, 12b     -> 12  (OM/intermediate cluster; heavy bidirectional confusion)
# 14a, 14b     -> 14  (Left PL cluster; naming convention + confusion)
# 15           -> 14  (LOW CONFIDENCE: not a lettered variant of 14, it's a
#                      distinct named vessel (left-dominant PDA) with its own
#                      clinical/SYNTAX-score meaning; only justified by a
#                      66.5% empirical confusion with 14b on this dataset.
#                      Reconsider before relying on this one.)
SEGMENT_MERGE_MAP = {
    9: 9, 10: 9, 11: 9, 12: 9,        # ids for codes 9, 9a, 10, 10a -> 9
    20: 20, 21: 20, 22: 20, 23: 20,   # ids for codes 16, 16a, 16b, 16c -> 16
    14: 14, 15: 14, 24: 14,           # ids for codes 12, 12a, 12b -> 12
    17: 17, 18: 17, 25: 17, 19: 17,   # ids for codes 14, 14a, 14b, 15 -> 14
}

# Raw ids that survive as their own merged class, in ascending order. This
# fixes the compact 1..14 numbering used by the merged label space.
_MERGED_ANCHOR_ORDER = sorted(set(SEGMENT_MERGE_MAP.get(i, i) for i in range(1, NUM_ANATOMY_CLASSES)))
MERGED_NUM_ANATOMY_CLASSES = len(_MERGED_ANCHOR_ORDER) + 1  # +1 background = 15

# raw id (0-25) -> merged id (0-14)
RAW_TO_MERGED_ID = [0] * NUM_ANATOMY_CLASSES
for _new_id, _raw_anchor in enumerate(_MERGED_ANCHOR_ORDER, start=1):
    RAW_TO_MERGED_ID[_raw_anchor] = _new_id
for _raw_id in range(1, NUM_ANATOMY_CLASSES):
    _raw_anchor = SEGMENT_MERGE_MAP.get(_raw_id, _raw_id)
    RAW_TO_MERGED_ID[_raw_id] = RAW_TO_MERGED_ID[_raw_anchor]


def remap_segment_id(segment_id: int) -> int:
    """Collapse a raw 0-25 SYNTAX segment id to its merged-class id (0-14)."""
    return RAW_TO_MERGED_ID[int(segment_id)]


# merged id -> metadata of the anchor raw segment it represents
MERGED_SEGMENT_LABELS = {
    _new_id: SYNTAX_SEGMENTS[_raw_anchor]
    for _new_id, _raw_anchor in enumerate(_MERGED_ANCHOR_ORDER, start=1)
}


def merged_segment_label(merged_id: int) -> str:
    meta = MERGED_SEGMENT_LABELS.get(int(merged_id))
    return "unknown" if meta is None else f"{meta['code']} {meta['name']}"


def merged_segment_group(merged_id: int) -> str:
    meta = MERGED_SEGMENT_LABELS.get(int(merged_id))
    return "unknown" if meta is None else meta["group"]


def merged_segment_artery(merged_id: int) -> str:
    meta = MERGED_SEGMENT_LABELS.get(int(merged_id))
    return "unknown" if meta is None else meta["artery"]


MERGED_SEGMENT_TO_GROUP_ID = [0] * MERGED_NUM_ANATOMY_CLASSES
MERGED_SEGMENT_TO_ARTERY_ID = [0] * MERGED_NUM_ANATOMY_CLASSES
for _new_id, _meta in MERGED_SEGMENT_LABELS.items():
    MERGED_SEGMENT_TO_GROUP_ID[_new_id] = GROUP_TO_ID[_meta["group"]]
    MERGED_SEGMENT_TO_ARTERY_ID[_new_id] = ARTERY_TO_ID[_meta["artery"]]
