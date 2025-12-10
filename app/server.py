# app/server.py

import datetime

from bson import ObjectId
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient

from .config import (
    DB_NAME,
    ITEMS_COLLECTION,
    MONGO_URI,
    SETTINGS_COLLECTION,
)
from .item_lookup import compute_adjusted_shelf_life
from .model_utils import load_model, predict_from_data_url

app = Flask(__name__, template_folder="templates", static_folder="static")

# --------- MongoDB setup ---------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
items_col = db[ITEMS_COLLECTION]
settings_col = db[SETTINGS_COLLECTION]

# --------- Load ML model once ---------
model = load_model()


# --------- Helpers for settings ---------
def get_current_temp(default=None):
    doc = settings_col.find_one({"_id": "current_temp"})
    if not doc:
        return default
    return doc.get("setTemp", default)


def set_current_temp(temp_value: float):
    settings_col.update_one(
        {"_id": "current_temp"},
        {"$set": {"setTemp": temp_value, "updatedAt": datetime.datetime.utcnow()}},
        upsert=True,
    )


def recompute_all_items_shelf_life(new_temp: float):
    """
    When temperature changes, recompute adjustedShelfLife
    for all existing items based on the new temperature.
    """
    cursor = items_col.find({})
    for doc in cursor:
        item_name = doc.get("itemName")
        if not item_name:
            continue
        rec_temp, base_shelf, adjusted = compute_adjusted_shelf_life(
            item_name, new_temp
        )
        if rec_temp is None:
            # Unknown item in lookup, skip
            continue

        items_col.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "setTemp": new_temp,
                    "recTemp": rec_temp,
                    "baseShelfLife": base_shelf,
                    "adjustedShelfLife": adjusted,
                }
            },
        )


# --------- Routes ---------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        data_url = data["image"]
        item, conf = predict_from_data_url(model, data_url)
        return jsonify({"item": item, "confidence": conf})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/set_temp", methods=["POST"])
def set_temp():
    data = request.get_json()
    if not data or "temperature" not in data:
        return jsonify({"error": "No temperature provided"}), 400

    try:
        temp = float(data["temperature"])
    except ValueError:
        return jsonify({"error": "Temperature must be a number"}), 400

    # Save new temp
    set_current_temp(temp)
    # Recompute shelf life for all existing items
    recompute_all_items_shelf_life(temp)

    return jsonify({"status": "ok", "setTemp": temp})


@app.route("/get_temp", methods=["GET"])
def get_temp():
    current = get_current_temp()
    return jsonify({"setTemp": current})


@app.route("/add_item", methods=["POST"])
def add_item():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    item = data.get("item", "").strip()
    qty = data.get("quantity", "").strip()

    if not item:
        return jsonify({"error": "Item is required"}), 400
    if not qty:
        return jsonify({"error": "Quantity is required"}), 400

    try:
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Quantity must be a positive integer"}), 400

    # Get current refrigerator temp
    current_temp = get_current_temp()
    if current_temp is None:
        return jsonify({"error": "Please set the refrigerator temperature first."}), 400

    # Compute shelf life based on lookup + current temp
    rec_temp, base_shelf, adjusted = compute_adjusted_shelf_life(item, current_temp)
    if rec_temp is None:
        return jsonify({"error": f"Item '{item}' not found in lookup."}), 400

    ts = datetime.datetime.utcnow()

    doc = {
        "itemName": item,
        "quantity": qty_int,
        "setTemp": current_temp,
        "recTemp": rec_temp,
        "baseShelfLife": base_shelf,
        "adjustedShelfLife": adjusted,
        "timestamp": ts,
    }

    items_col.insert_one(doc)

    return jsonify({"status": "ok"})


@app.route("/items", methods=["GET"])
def get_items():
    docs = list(items_col.find().sort("timestamp", -1))
    items = []

    now = datetime.datetime.utcnow()

    for d in docs:
        ts = d.get("timestamp")
        total = d.get(
            "adjustedShelfLife"
        )  # total shelf life in seconds at current temp

        if ts is not None and total is not None:
            elapsed = (now - ts).total_seconds()
            remaining = int(round(total - elapsed))
            if remaining < 0:
                remaining = 0
        else:
            remaining = None

        items.append(
            {
                "id": str(d.get("_id")),
                "item": d.get("itemName"),
                "quantity": d.get("quantity"),
                "setTemp": d.get("setTemp"),
                "recTemp": d.get("recTemp"),
                "baseShelfLife": d.get("baseShelfLife"),
                "adjustedShelfLife": total,  # for future alerts/analytics
                "shelfLife": remaining,  # 👈 this is what the UI uses now
                "timestamp": ts.isoformat() if ts else None,
            }
        )

    return jsonify({"items": items})


@app.route("/use_item", methods=["POST"])
def use_item():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    item_id = data.get("id")
    used_qty = data.get("usedQuantity")

    if not item_id:
        return jsonify({"error": "Item ID is required"}), 400
    if used_qty is None:
        return jsonify({"error": "usedQuantity is required"}), 400

    try:
        used_qty = int(used_qty)
        if used_qty <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "usedQuantity must be a positive integer"}), 400

    try:
        oid = ObjectId(item_id)
    except Exception:
        return jsonify({"error": "Invalid item ID"}), 400

    doc = items_col.find_one({"_id": oid})
    if not doc:
        return jsonify({"error": "Item not found"}), 404

    current_qty = doc.get("quantity", 0)
    if used_qty >= current_qty:
        # All (or more) used -> remove item completely
        items_col.delete_one({"_id": oid})
        return jsonify({"status": "deleted", "remainingQuantity": 0})
    else:
        # Partially used -> decrement quantity
        new_qty = current_qty - used_qty
        items_col.update_one({"_id": oid}, {"$set": {"quantity": new_qty}})
        return jsonify({"status": "updated", "remainingQuantity": new_qty})
