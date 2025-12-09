# app/server.py

import csv
import datetime
import os

from flask import Flask, jsonify, render_template, request

from .config import DATA_CSV_PATH
from .model_utils import load_model, predict_from_data_url

# Create Flask app, with templates & static under app/
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model once at startup
model = load_model()


def append_to_csv(item: str, quantity: int, timestamp: str):
    # Check if file is empty to decide whether to write header
    write_header = not os.path.exists(DATA_CSV_PATH) or os.path.getsize(DATA_CSV_PATH) == 0
    
    with open(DATA_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["item", "quantity", "timestamp"])
        writer.writerow([item, quantity, timestamp])


def read_items_from_csv():
    items = []
    if not os.path.exists(DATA_CSV_PATH):
        return items

    with open(DATA_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            first_row = next(reader)
            if first_row != ['item', 'quantity', 'timestamp']:
                # This is not a header, so it's data
                items.append({'item': first_row[0], 'quantity': first_row[1], 'timestamp': first_row[2]})
        except StopIteration:
            return [] # Empty file

        for row in reader:
            items.append({'item': row[0], 'quantity': row[1], 'timestamp': row[2]})
            
    return items


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

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_to_csv(item, qty_int, ts)

    return jsonify({"status": "ok"})


@app.route("/items", methods=["GET"])
def get_items():
    items = read_items_from_csv()
    return jsonify({"items": items})
