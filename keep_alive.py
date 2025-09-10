"""
Keep Alive Server for Telegram Bot
Maintains bot activity to prevent automatic sleep
"""

from flask import Flask, jsonify
from threading import Thread
import logging
import datetime

# Configure logging for keep-alive
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    """Health check endpoint"""
    return "Bot is alive!"

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "uptime": "active",
        "services": {
            "telegram_bot": "running",
            "gpt4o": "connected",
            "gemini": "connected"
        }
    })

@app.route('/status')
def status():
    """Bot status endpoint"""
    return jsonify({
        "bot_name": "Iraqi AI Bot",
        "models": ["GPT-4o", "Gemini 2.0 Flash"],
        "features": ["Chat", "Image Generation", "Translation", "Image Analysis"],
        "language": "Iraqi Arabic"
    })

def run_server():
    """Run the Flask server on port 8080"""
    try:
        # Bind to 0.0.0.0:8080 for external monitoring
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting keep-alive server: {e}")

def keep_alive():
    """Start the keep-alive server in a separate thread"""
    try:
        logger.info("Starting keep-alive server on port 8080...")
        server_thread = Thread(target=run_server)
        server_thread.daemon = True  # Dies when main thread dies
        server_thread.start()
        logger.info("Keep-alive server started successfully!")
    except Exception as e:
        logger.error(f"Failed to start keep-alive server: {e}")

if __name__ == "__main__":
    keep_alive()