# app/config.py

import os

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
