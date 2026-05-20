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
