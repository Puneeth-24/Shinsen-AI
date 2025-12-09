# app/item_lookup.py

import json

from .config import ITEMS_LOOKUP_PATH

with open(ITEMS_LOOKUP_PATH, "r", encoding="utf-8") as f:
    _ITEMS = json.load(f)


def get_item_params(item_name: str):
    """
    Returns a dict like {"storeTemp": 4, "shelfLife": 45} or None
    """
    return _ITEMS.get(item_name)


def compute_adjusted_shelf_life(item_name: str, fridge_temp: float):
    """
    Uses rules:
      - baseShelfLife = shelfLife from JSON (seconds) @ recommended storeTemp
      - delta = fridge_temp - recommended_temp
      - for each +1°C above: shelf life * 0.9 (–10% per degree)
      - for each –1°C below: shelf life * 1.05 (+5% per degree)
    Returns (recommended_temp, base_shelf_life, adjusted_shelf_life).
    adjusted_shelf_life is int (rounded).
    """
    params = get_item_params(item_name)
    if not params:
        return None, None, None

    rec_temp = float(params["storeTemp"])
    base_shelf = float(params["shelfLife"])  # seconds
    delta = fridge_temp - rec_temp

    if delta > 0:
        # Above recommended: –10% per degree
        factor = 0.9**delta
    elif delta < 0:
        # Below recommended: +5% per degree
        factor = 1.05 ** (-delta)
    else:
        factor = 1.0

    adjusted = int(round(base_shelf * factor))

    return rec_temp, int(base_shelf), adjusted
