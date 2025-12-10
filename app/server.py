import atexit
import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from bson import ObjectId
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from twilio.rest import Client  # 👈 NEW

from .config import (
    ALERT_THRESHOLD_SECONDS,
    ALERT_WHATSAPP_TO,
    DB_NAME,
    ITEMS_COLLECTION,
    MONGO_URI,
    SETTINGS_COLLECTION,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
)
from .item_lookup import compute_adjusted_shelf_life
from .model_utils import load_model, predict_from_data_url

app = Flask(__name__, template_folder="templates", static_folder="static")

# --------- MongoDB setup ---------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
items_col = db[ITEMS_COLLECTION]
settings_col = db[SETTINGS_COLLECTION]

# --------- Twilio client ---------
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
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


def compute_remaining_shelf_life(doc, now=None):
    """
    Compute remaining shelf life (in seconds) for a DB document.
    Uses fields:
      - timestamp (datetime)
      - adjustedShelfLife (total shelf life in seconds at current temp)

    Returns an integer (>= 0) or None if not computable.
    """
    if now is None:
        now = datetime.datetime.utcnow()

    ts = doc.get("timestamp")
    total = doc.get("adjustedShelfLife")
    if ts is None or total is None:
        return None

    elapsed = (now - ts).total_seconds()
    remaining = int(round(total - elapsed))
    if remaining < 0:
        remaining = 0

    return remaining


def get_expiring_items(threshold_seconds: int):
    """
    Returns a list of (doc, remaining_seconds) for items
    whose remaining shelf life is > 0 and <= threshold_seconds
    AND that have not been alerted before (alertSent != True).
    """
    now = datetime.datetime.utcnow()
    docs = items_col.find({"alertSent": {"$ne": True}})
    expiring = []

    for d in docs:
        remaining = compute_remaining_shelf_life(d, now)
        if remaining is None:
            continue
        if 0 < remaining <= threshold_seconds:
            expiring.append((d, remaining))

    expiring.sort(key=lambda x: x[1])  # soonest first
    return expiring


def format_alert_message(expiring_items, current_temp, threshold_seconds):
    """
    Build a single WhatsApp message body listing all expiring items.
    """
    if not expiring_items:
        return None

    lines = []
    lines.append("🚨 *Shinsen-AI Fridge Alert*")
    lines.append("")
    lines.append(
        f"The following items are close to expiring (≤ {threshold_seconds} seconds remaining):"
    )
    lines.append("")

    for doc, remaining in expiring_items:
        name = doc.get("itemName", "Unknown item")
        qty = doc.get("quantity", "?")
        rec_temp = doc.get("recTemp")
        base = doc.get("baseShelfLife")
        adj = doc.get("adjustedShelfLife")

        line = f"• *{name}* – qty: {qty}, remaining: {remaining}s"
        if rec_temp is not None:
            line += f", rec. temp: {rec_temp}°C"
        if adj is not None and base is not None:
            line += f" (base: {base}s, adjusted: {adj}s)"
        lines.append(line)

    lines.append("")
    if current_temp is not None:
        lines.append(f"Current fridge temperature: *{current_temp}°C*")
    lines.append("")
    lines.append("ℹ️ Consider using these items soon to avoid spoilage.")

    return "\n".join(lines)


def send_whatsapp_alert(message_body: str):
    """
    Sends a WhatsApp alert using Twilio.
    Returns the Twilio message SID.
    """
    msg = twilio_client.messages.create(
        body=message_body,
        from_=TWILIO_WHATSAPP_FROM,
        to=ALERT_WHATSAPP_TO,
    )
    return msg.sid


def process_alerts(threshold_seconds: int | None = None, mark_alerted: bool = True):
    """
    Core alert logic:
      - find expiring items
      - format WhatsApp message
      - send via Twilio
      - mark them as alerted (alertSent=true) if required
    Returns a dict result summary.
    """
    if threshold_seconds is None:
        threshold_seconds = ALERT_THRESHOLD_SECONDS

    expiring = get_expiring_items(threshold_seconds)
    if not expiring:
        return {"status": "no_items_expiring", "threshold": threshold_seconds}

    current_temp = get_current_temp()
    message_body = format_alert_message(expiring, current_temp, threshold_seconds)
    if not message_body:
        return {"status": "no_message_built"}

    sid = send_whatsapp_alert(message_body)

    # Mark these items as alerted so we don't spam
    if mark_alerted:
        now = datetime.datetime.utcnow()
        ids = [doc["_id"] for (doc, _) in expiring]
        items_col.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"alertSent": True, "alertSentAt": now}},
        )

    return {
        "status": "sent",
        "threshold": threshold_seconds,
        "num_items": len(expiring),
        "twilio_sid": sid,
    }


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
        remaining = compute_remaining_shelf_life(d, now)

        items.append(
            {
                "id": str(d.get("_id")),
                "item": d.get("itemName"),
                "quantity": d.get("quantity"),
                "setTemp": d.get("setTemp"),
                "recTemp": d.get("recTemp"),
                "baseShelfLife": d.get("baseShelfLife"),
                "adjustedShelfLife": d.get("adjustedShelfLife"),
                "shelfLife": remaining,
                "timestamp": d.get("timestamp").isoformat()
                if d.get("timestamp")
                else None,
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


@app.route("/send_alerts", methods=["POST"])
def send_alerts():
    """
    Manual trigger of the alert process.
    Useful for debugging or a 'Send alerts now' button.
    """
    try:
        result = process_alerts()
        return jsonify(result)
    except Exception as e:
        print("send_alerts error:", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# --------- APScheduler: periodic alert checks ---------
scheduler = BackgroundScheduler()


def scheduled_check_alerts():
    """
    Background job that periodically checks for expiring items
    and sends a grouped WhatsApp alert once per item.
    """
    with app.app_context():
        try:
            result = process_alerts()
            # Optional: log result to console
            print("[APScheduler] Alert check:", result)
        except Exception as e:
            print("[APScheduler] Error in scheduled alert job:", e)


# Run every 5 seconds (you can tweak this)
scheduler.add_job(scheduled_check_alerts, "interval", seconds=5)
scheduler.start()

# Shut down scheduler when app exits
atexit.register(lambda: scheduler.shutdown())
