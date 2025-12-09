import datetime

from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient

from .config import (
    DB_NAME,
    ITEMS_COLLECTION,
    MONGO_URI,
    SETTINGS_COLLECTION,
)
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

    set_current_temp(temp)
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

    # Get current refrigerator temp (can be None if not set yet)
    current_temp = get_current_temp()

    ts = datetime.datetime.utcnow()

    doc = {
        "itemName": item,
        "quantity": qty_int,
        "setTemp": current_temp,  # <-- important part
        "timestamp": ts,
    }

    items_col.insert_one(doc)

    return jsonify({"status": "ok"})


@app.route("/items", methods=["GET"])
def get_items():
    docs = list(items_col.find().sort("timestamp", -1))
    items = []
    for d in docs:
        items.append(
            {
                "id": str(d.get("_id")),
                "item": d.get("itemName"),
                "quantity": d.get("quantity"),
                "setTemp": d.get("setTemp"),
                "timestamp": d.get("timestamp").isoformat()
                if d.get("timestamp")
                else None,
            }
        )
    return jsonify({"items": items})
