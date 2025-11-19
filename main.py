import cv2
import numpy as np
from tensorflow import keras

# -----------------------------
# 1. Config
# -----------------------------
MODEL_PATH = "model/mobilenetv2_fruits_veggies_finetuned.h5"  # change if needed
IMG_SIZE = (224, 224)  # must match what you used for training

# Your 36 class names (from dataset folders)
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

# -----------------------------
# 2. Load model
# -----------------------------
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded!")


# -----------------------------
# 3. Helper: preprocess frame
# -----------------------------
def preprocess_frame(frame):
    """
    Takes a BGR image from OpenCV, converts to RGB,
    resizes to IMG_SIZE, and prepares batch of 1 for prediction.
    """
    # OpenCV gives BGR, convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    # Convert to float32
    img_array = img_resized.astype("float32")

    # IMPORTANT:
    # If you used layers.Rescaling(1./255) INSIDE the model (as in our code),
    # you DO NOT need to divide by 255 here.
    # If you did NOT, then uncomment the next line:
    # img_array /= 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# -----------------------------
# 4. Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    img_batch = preprocess_frame(frame)

    # Predict
    preds = model.predict(img_batch, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_class = CLASS_NAMES[pred_idx]
    confidence = preds[0][pred_idx]

    # -------------------------
    # 5. Display prediction
    # -------------------------
    label_text = f"{pred_class} ({confidence * 100:.1f}%)"

    # Put label on frame
    cv2.putText(
        frame,
        label_text,
        (10, 30),  # position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,  # font scale
        (0, 255, 0),  # color (B, G, R)
        2,
        cv2.LINE_AA,
    )

    # Show the frame
    cv2.imshow("Fruit & Veg Detector", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
