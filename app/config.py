# app/config.py

import os

from dotenv import load_dotenv

# Base directory = project root (fruit_veg_webapp/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to your trained model
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_fruits_veggies_finetuned.h5")

# Path to CSV where we log items + quantity
# DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "items_log.csv")

# Make sure data dir exists
# os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)

# Image size used during training
IMG_SIZE = (224, 224)

# Class names (order MUST match training)
CLASS_NAMES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]
# ----------------MongoDb----------------
MONGO_URI = "mongodb://localhost:27017"  # change if using Atlas
DB_NAME = "fruit_veg_db"
ITEMS_COLLECTION = "items_log"
SETTINGS_COLLECTION = "settings"
SCANNED_ITEMS = "scanned_items"
# -----------------Item Lookup JSON---------
ITEMS_LOOKUP_PATH = os.path.join(BASE_DIR, "data/items_lookup.json")


# ---------- Twilio / WhatsApp Alert Config ----------
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# From number: Twilio WhatsApp sandbox or your WhatsApp-enabled number
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")

# To number: your WhatsApp phone (must be joined to sandbox / approved)
ALERT_WHATSAPP_TO = os.getenv("ALERT_WHATSAPP_TO")

# Items with remaining shelf life <= this many seconds will be considered "expiring soon"
ALERT_THRESHOLD_SECONDS = int(os.getenv("ALERT_THRESHOLD_SECONDS", "15"))
