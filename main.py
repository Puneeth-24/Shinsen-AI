# main.py

from app.server import app

if __name__ == "__main__":
    # For development
    app.run(host="0.0.0.0", port=5000, debug=True)
