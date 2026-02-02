# Shinsen-AI   
**AI-Powered Smart Refrigerator for Food Recognition, Inventory & Freshness Management**

Shinsen-AI is an intelligent **Smart Refrigerator system** that uses Artificial Intelligence and a database backend to automatically recognize, track, and manage food items placed inside a refrigerator. The system combines computer vision, a Python backend, and a web frontend to help users reduce food waste, monitor inventory, and manage expiry information.

---

##  Project Overview

Traditional refrigerators offer no insights into food contents, leading to forgotten items and unnecessary waste. **Shinsen-AI** brings intelligence to your refrigerator by:

- Automatically detecting and classifying food items using deep learning
- Storing and organizing inventory data in a database
- Offering a web interface for checking available items
- Serving as a platform for future smart alert systems and predictions

---

## Key Features

-  **Computer Vision Food Recognition** using fine-tuned models
-  **Inventory Management with MongoDB**
-  **Model Training & Experimentation Notebooks**
-  **Web Frontend built with Node.js + Tailwind CSS**
-  **Modular Architecture for Feature Growth**

---

## Project Structure
```bash
├── app
│   ├── config.py
│   ├── __init__.py
│   ├── item_lookup.py
│   ├── model_utils.py
│   ├── __pycache__
│   ├── server.py
│   ├── static
│   └── templates
├── data
│   ├── items_log.csv
│   └── items_lookup.json
├── main.py
├── mobilenetv2-fine-tuning.ipynb
├── models
│   └── mobilenetv2_fruits_veggies_finetuned.h5
├── package.json
├── package-lock.json
├── README.md
├── requirements.txt
└── tailwind.config.js
```

---

## Tech Stack

**Backend**
- Python (Flask or FastAPI compatible)
- MongoDB (Database for storing item info and inventory)
- TensorFlow / Keras for model inference

**Frontend**
- Node.js
- Tailwind CSS
- Vanilla JavaScript or framework (depending on implementation)

---

## Prerequisites

Before running the project, make sure you have the following installed:

| Requirement | Version |
|-------------|---------|
| Python       | 3.8+    |
| pip          | Latest  |
| Node.js      | Latest LTS |
| MongoDB      | Running locally or Atlas |
| Git          | Latest  |

---

## Setup & Installation

1. Clone the Repository

```bash
git clone https://github.com/Puneeth-24/Shinsen-AI.git
cd Shinsen-AI
```

2. MongoDB Setup

You can use either:
- MongoDB Atlas (cloud)
- Local MongoDB server

Get your connection string — it should look like:
```bash 
mongodb+srv://<username>:<password>@cluster0.mongodb.net/shinsenAI?retryWrites=true&w=majority
```
3. Environment Variables

Create a .env file in the root directory:
```bash
touch .env
```
Add the following:
```bash
MONGODB_URI = your_mongodb_connection_string
PORT = 5000
```
Replace your_mongodb_connection_string with the connection string from the previous step.

4. Install Python Dependencies

Optional — create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```
Install requirements:
```bash
pip install -r requirements.txt
```

5. Install Frontend Dependencies
```bash
npm install
```
This installs all node packages required for the UI.

---

## Running the Project

1. Start the Backend
```bash
python main.py
```
This will:
- Start the Python backend
- Connect to your MongoDB database
- Load the AI models for item recognition

2. Start the Frontend
```bash
npm run dev
```
Visit the displayed local URL (e.g., http://localhost:3000) in your browser to interact with the interface.

---

## Model Training & Data

To experiment with or retrain models:
1. Open mobilenetv2-fine-tuning.ipynb
2. Use images in the data/ directory
3. Train or fine-tune the model
4. Save the final model to models/
5. Restart the backend to load the updated model

---

## Backend & Database Workings

The backend is responsible for:
- Accepting image uploads from the frontend
- Running AI inference on food items
- Storing recognized item data (name, category, timestamp) in MongoDB
- Providing API endpoints for fetching inventory

All item metadata and inventory states are stored in the MongoDB database configured via MONGODB_URI.

---

## Future Enhancements

Here are potential next steps for Shinsen-AI:
- Expiry date prediction and push notifications
- Nutritional analysis & consumption statistics
- Smart grocery suggestions based on patterns
- Mobile app for remote fridge monitoring
- Cloud deployment & real-time sync
