from flask import Flask, render_template, request
from flask import session, redirect, url_for
import tensorflow as tf
import numpy as np
import time 
from PIL import Image
import os
import requests
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import mysql.connector
from werkzeug.security import generate_password_hash,check_password_hash
from translation import TRANSLATIONS

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY if not configured")
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)
@app.context_processor
def inject_translations():
    lang = session.get("language", "en")
    return {
        "t": TRANSLATIONS.get(lang, TRANSLATIONS["en"]),
        "current_language": lang
    }


@app.route("/set-language/<lang>")
def set_language(lang):
    if lang in ("en", "mr"):
        session["language"] = lang

    return redirect(request.referrer or url_for("dashboard"))

# =====================================
# DETECTION HISTORY DATABASE
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    "detection_history.db"
)


def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            disease TEXT NOT NULL,
            confidence REAL,
            detected_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()
def save_detection(image_name, disease, confidence):
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO detection_history
        (image_name, disease, confidence, detected_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            image_name,
            disease,
            confidence,
            datetime.now().strftime("%d-%m-%Y %I:%M %p")
        )
    )

    conn.commit()
    conn.close()

MODEL_PATH = "tomato_disease_model.keras"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf mold",
    "Mosaic virus",
    "Septoria leaf spot",
    "Spider mite",
    "Target spot",
    "Yellow leaf curl virus",
    "Healthy"
]

disease_info = {
    "Bacterial spot": {
        "symptoms": "Small dark brown or black spots appear on leaves.",
        "treatment": "Remove infected leaves and avoid overhead watering.",
        "prevention": "Use healthy seeds and maintain good spacing."
    },

    "Early blight": {
        "symptoms": "Brown circular spots with ring-like patterns appear on older leaves.",
        "treatment": "Remove infected leaves and improve air circulation.",
        "prevention": "Avoid overhead watering and maintain proper spacing."
    },

    "Late blight": {
        "symptoms": "Dark green to brown irregular spots appear on leaves.",
        "treatment": "Remove infected leaves and avoid wet foliage.",
        "prevention": "Maintain good air circulation and avoid overhead irrigation."
    },

    "Leaf mold": {
        "symptoms": "Yellow spots appear on leaves and fungal growth may appear underneath.",
        "treatment": "Remove infected leaves and improve ventilation.",
        "prevention": "Reduce humidity and avoid excessive watering."
    },

    "Mosaic virus": {
        "symptoms": "Leaves show light and dark green mosaic patterns.",
        "treatment": "Remove badly infected plants and control insect vectors.",
        "prevention": "Use healthy planting material and control insects."
    },

    "Septoria leaf spot": {
        "symptoms": "Small circular spots with dark borders appear on leaves.",
        "treatment": "Remove infected leaves and improve air circulation.",
        "prevention": "Avoid overhead watering and maintain plant spacing."
    },

    "Spider mite": {
        "symptoms": "Tiny yellow spots, leaf drying and webbing may occur.",
        "treatment": "Monitor plants and control mites using suitable methods.",
        "prevention": "Check leaves regularly and maintain plant health."
    },

    "Target spot": {
        "symptoms": "Brown circular spots with target-like rings appear on leaves.",
        "treatment": "Remove infected plant material and improve air circulation.",
        "prevention": "Avoid excess moisture and maintain proper spacing."
    },

    "Yellow leaf curl virus": {
        "symptoms": "Leaves become yellow and curl upward.",
        "treatment": "Control whiteflies and remove severely infected plants.",
        "prevention": "Use healthy seedlings and control whiteflies."
    },

    "Healthy": {
        "symptoms": "The leaf appears healthy without major visible disease symptoms.",
        "treatment": "No disease treatment is required.",
        "prevention": "Provide proper sunlight, water and nutrients."
    }
}


def get_weather(city):

    if not OPENWEATHER_API_KEY:
        print("OPENWEATHER_API_KEY not found in .env")
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:
            print("Weather API Error:", data)
            return None

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }

    except Exception as e:
        print("Weather Error:", e)
        return None

@app.before_request
def require_login():
    allowed_endpoints = {"login", "register", "static"}

    if request.endpoint is None:
        return

    if request.endpoint in allowed_endpoints:
        return

    if "user_id" not in session:
        return redirect(url_for("login"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = None
        cursor = None

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, name, password_hash FROM users WHERE email = %s",
                (email,)
            )
            user = cursor.fetchone()

            if user and check_password_hash(user[2], password):
                session.clear()
                session["user_id"] = user[0]
                session["user_name"] = user[1]
                return redirect(url_for("dashboard"))

            return "Invalid email or password.", 401

        except mysql.connector.Error:
            app.logger.exception("Login database error")
            return "Database error. Please check the Flask terminal.", 500

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    return render_template("login.html")

# =====================================
# MYSQL DATABASE CONNECTION
# =====================================

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        ssl_disabled=False
    )
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            return "Name, email and password are required.", 400

        if password != confirm_password:
            return "Passwords do not match.", 400

        if len(password) < 8:
            return "Password must contain at least 8 characters.", 400

        conn = None
        cursor = None

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            password_hash = generate_password_hash(password)

            cursor.execute(
                """
                INSERT INTO users
                (name, email, mobile, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (name, email, mobile, password_hash)
            )

            conn.commit()
            return "Registration successful! You can now log in."

        except mysql.connector.IntegrityError:
            return "This email is already registered.", 409

        except mysql.connector.Error:
            app.logger.exception("Registration database error")
            return "Database error. Please check the Flask terminal.", 500

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    return render_template("register.html")

def save_chat_history(question, answer):
    conn = None
    cursor = None

    try:
        if "user_id" not in session:
            return

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history (user_id, question, answer)
            VALUES (%s, %s, %s)
            """,
            (session["user_id"], question, answer)
        )

        conn.commit()

    except mysql.connector.Error:
        app.logger.exception("Could not save chat history")

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()

@app.after_request
def record_ai_chat(response):
    if (
        request.path == "/ask-ai"
        and request.method == "POST"
        and response.status_code == 200
        and "user_id" in session
    ):
        question = request.form.get("question", "").strip()
        answer = response.get_data(as_text=True)

        if (
            question
            and answer
            and not answer.lstrip().lower().startswith("<!doctype html")
            and not answer.lstrip().lower().startswith("<html")
        ):
            save_chat_history(question, answer)

    return response
@app.route("/ai-assistant")
def ai_assistant():

    disease = request.args.get(
        "disease",
        ""
    )

    return render_template(
        "ai_assistant.html",
        disease=disease
    )


@app.route("/ask-ai", methods=["POST"])
def ask_ai():

    question = request.form.get(
        "question",
        ""
    ).strip()

    question_lower = question.lower()


    # WEATHER
    weather_words = [
        "weather",
        "temperature",
        "rain",
        "rainfall",
        "climate",
        "humidity",
        "forecast",
        "wind",
        "पाऊस",
        "हवामान",
        "तापमान"
    ]


    if any(
        word in question_lower
        for word in weather_words
    ):

        cities = [
            "Pune",
            "Mumbai",
            "Nashik",
            "Nagpur",
            "Kolhapur",
            "Satara",
            "Solapur",
            "Sangli",
            "Ahmednagar",
            "Pandharpur",
            "Kudal",
            "Delhi",
            "Bangalore",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Ahmedabad",
            "Jaipur",
            "Indore"
        ]

        selected_city = None

        for city in cities:

            if city.lower() in question_lower:
                selected_city = city
                break


        if selected_city is None:

            return (
                "🌦️ I can provide live weather information.\n\n"
                "Please mention your city name.\n\n"
                "Examples:\n"
                "• What is the weather in Pune?\n"
                "• Temperature in Mumbai?\n"
                "• Is it raining in Nashik?"
            )


        weather = get_weather(
            selected_city
        )


        if weather is None:

            return (
                "⚠️ Sorry, I could not get live weather "
                "information right now.\n\n"
                "Please check your internet connection "
                "or try again."
            )


        return (
            f"🌦️ Weather in {weather['city']}\n\n"
            f"🌡️ Temperature: "
            f"{weather['temperature']} °C\n\n"
            f"🤗 Feels like: "
            f"{weather['feels_like']} °C\n\n"
            f"💧 Humidity: "
            f"{weather['humidity']}%\n\n"
            f"☁️ Condition: "
            f"{weather['condition'].title()}\n\n"
            f"💨 Wind speed: "
            f"{weather['wind_speed']} m/s\n\n"
            "🌱 Farming Tip:\n"
            "Monitor soil moisture and crop condition "
            "according to the current weather."
        )
        # CROP PLANNING AND SEED RECOMMENDATION
    crop_words = [
        "crop", "crops", "farming", "sowing", "sow",
        "seed", "seeds", "variety", "which farming",
        "which crop", "पेरणी", "बियाणे", "बी", "शेती",
        "कोणते पीक", "पीक कोणते"
    ]

    if any(word in question_lower for word in crop_words):
        # SOYBEAN SEED VARIETIES
        if any(word in question_lower for word in [
            "soybean", "soyabean", "soyabeans", "सोयाबीन"
        ]):
            return (
                "🌱 Soybean Seed Variety Guide - Maharashtra\n\n"
                "Varieties to check for regional suitability:\n"
                "• MAUS 725\n"
                "• Phule Durva (KDS 992)\n"
                "• Phule Kimaya (KDS 753)\n"
                "• Phule Sangam (KDS 726)\n"
                "• MAUS 731\n\n"
                "These varieties are reported in recommendations "
                "for parts of Maharashtra. Suitability differs by region.\n\n"
                "Sowing season: Soybean is generally a Kharif crop, "
                "sown after suitable monsoon rainfall, commonly in June-July.\n\n"
                "Please tell me your district and available irrigation "
                "so I can help narrow down the options.\n\n"
                "Important: Confirm the current district-specific "
                "recommendation and certified seed availability with "
                "your local Krishi Vigyan Kendra before buying."
            )
            # OTHER CROP RECOMMENDATIONS

        # CHICKPEA / HARBHARA
        if any(word in question_lower for word in [
            "harbhara", "chana", "chickpea", "gram", "हरभरा", "चना"
        ]):
            return (
                "🌱 Harbhara (Chickpea) Guide\n\n"
                "Suggested varieties to check:\n"
                "• Phule Vikram\n"
                "• Phule Vikrant\n"
                "• JAKI-9218\n"
                "• Digvijay\n\n"
                "Season: Rabi (October-November is a common "
                "sowing period in Maharashtra).\n"
                "Use well-drained soil and avoid excess irrigation.\n"
                "Choose certified seeds suitable for your district.\n"
                "Confirm the latest recommendation with your local KVK."
            )

        # WHEAT
        if any(word in question_lower for word in [
            "wheat", "gahu", "गेहू", "गहू"
        ]):
            return (
                "🌾 Wheat Farming Guide\n\n"
                "Suggested varieties to check for Maharashtra:\n"
                "• NIAW-301 (Trimbak)\n"
                "• NIAW-917 (Tapovan)\n"
                "• NIAW-1994 (Phule Samadhan)\n"
                "• GW-496\n\n"
                "Season: Rabi. Sowing time depends on your district, "
                "temperature and irrigation availability.\n"
                "Timely sowing and suitable irrigation are important.\n"
                "Confirm local suitability before buying seeds."
            )

        # ONION / KANDA
        if any(word in question_lower for word in [
            "onion", "kanda", " कांदा", "कांदा"
        ]):
            return (
                "🧅 Onion Farming Guide\n\n"
                "Varieties to check:\n"
                "• Bhima Shakti\n"
                "• Bhima Red\n"
                "• Bhima Super\n"
                "• Baswant-780\n"
                "• N-2-4-1\n\n"
                "Onion can be grown in different seasons in Maharashtra.\n"
                "For Rabi onion, nursery sowing is commonly done "
                "around October-November and transplanting around "
                "December-January.\n"
                "Choose the variety according to season, soil and "
                "market needs. Check local recommendations."
            )

        # MAIZE / MAKKA
        if any(word in question_lower for word in [
            "maize", "corn", "makka", "मका"
        ]):
            return (
                "🌽 Maize Farming Guide\n\n"
                "Maize varieties and hybrids depend on season "
                "and local conditions.\n"
                "• Select certified seed recommended for your district.\n"
                "• Check whether the variety suits Kharif or Rabi sowing.\n"
                "• Ensure proper drainage and moisture during growth.\n"
                "Tell me your district and irrigation availability "
                "for more specific guidance."
            )

        # TUR / PIGEON PEA
        if any(word in question_lower for word in [
            "tur", "pigeon pea", "arhar", "तूर", "अरहर"
        ]):
            return (
                "🌿 Tur (Pigeon Pea) Farming Guide\n\n"
                "Tur is mainly a Kharif crop in Maharashtra.\n"
                "Choose a locally recommended, certified variety "
                "based on crop duration and rainfall.\n"
                "Avoid waterlogging and monitor the crop for pests.\n"
                "Ask your local KVK for varieties suitable for your area."
            )

        # COTTON / KAPUS
        if any(word in question_lower for word in [
            "cotton", "kapus", "कापूस"
        ]):
            return (
                "🌱 Cotton Farming Guide\n\n"
                "Choose certified cotton seed approved for your "
                "region and cultivation conditions.\n"
                "Check the current approved hybrid list and seed label.\n"
                "Sowing time, spacing and irrigation depend on "
                "rainfall, soil and the selected hybrid.\n"
                "Consult your local agriculture officer before purchase."
            )

        # WINTER / RABI CROPS
        if any(word in question_lower for word in [
            "rabi", "winter", "रब्बी"
        ]):
            return (
                "🌾 Rabi Season Crop Guide (Maharashtra)\n\n"
                "Common crops:\n"
                "• Wheat\n"
                "• Chickpea (Gram)\n"
                "• Rabi Jowar\n"
                "• Safflower\n\n"
                "Sowing: Generally October to December, "
                "depending on crop, irrigation and local conditions.\n\n"
                "Seed selection:\n"
                "Choose a variety recommended for your district "
                "by the local Krishi Vigyan Kendra or agriculture university.\n\n"
                "Tell me your district, soil type and water availability "
                "for more suitable suggestions."
            )

        # SUMMER CROPS
        if any(word in question_lower for word in [
            "summer", "उन्हाळा", "उन्हाळी"
        ]):
            return (
                "☀️ Summer Crop Guide\n\n"
                "Possible crops where irrigation is available:\n"
                "• Summer Groundnut\n"
                "• Green Gram (Moong)\n"
                "• Vegetables suited to the local climate\n\n"
                "Sowing time and variety depend on temperature, "
                "water availability and local recommendations.\n\n"
                "Please tell me your district and irrigation availability."
            )

        # KHARIF / MONSOON CROPS
        if any(word in question_lower for word in [
            "kharif", "monsoon", "rainy", "खरीप", "पावसाळा"
        ]):
            return (
                "🌧️ Kharif Season Crop Guide (Maharashtra)\n\n"
                "Common crops:\n"
                "• Soybean\n"
                "• Cotton\n"
                "• Tur (Pigeon Pea)\n"
                "• Maize\n"
                "• Groundnut\n\n"
                "Sowing: Usually around the onset of the monsoon, "
                "after suitable soil moisture is available.\n\n"
                "Seed names and sowing dates must be selected "
                "according to your district and official recommendations.\n\n"
                "Tell me your district, soil type and water availability."
            )

        # GENERAL CROP PLANNING
        return (
            "🌱 Crop Planning Assistant\n\n"
            "I can guide you about:\n"
            "• Kharif crops: Soybean, Cotton, Tur, Maize\n"
            "• Rabi crops: Wheat, Chickpea, Rabi Jowar\n"
            "• Summer crops: Moong, Groundnut and suitable vegetables\n\n"
            "To recommend a suitable crop and seed variety, tell me:\n"
            "1. Your district\n"
            "2. Soil type (black, red or other)\n"
            "3. Water availability (rainfed or irrigation)\n"
            "4. Which month you plan to sow\n\n"
            "Important: Confirm the seed variety with local "
            "agriculture experts before purchasing."
        )


    # EARLY BLIGHT
    if "early blight" in question_lower:

        return (
            "🌿 Early Blight\n\n"
            "Symptoms:\n"
            "Brown circular spots with ring-like patterns "
            "usually appear on older leaves.\n\n"
            "Treatment:\n"
            "Remove infected leaves and improve air circulation.\n\n"
            "Prevention:\n"
            "Avoid overhead watering and maintain proper spacing."
        )


    # LATE BLIGHT
    if "late blight" in question_lower:

        return (
            "🌿 Late Blight\n\n"
            "Symptoms:\n"
            "Dark green or brown irregular spots can appear "
            "on leaves.\n\n"
            "Treatment:\n"
            "Remove severely infected leaves and avoid wet foliage.\n\n"
            "Prevention:\n"
            "Maintain good air circulation."
        )


    # BACTERIAL SPOT
    if "bacterial spot" in question_lower:

        return (
            "🦠 Bacterial Spot\n\n"
            "Symptoms:\n"
            "Small dark brown or black spots can appear on leaves.\n\n"
            "Treatment:\n"
            "Remove infected leaves and avoid overhead watering.\n\n"
            "Prevention:\n"
            "Use healthy seeds and maintain good spacing."
        )


    # LEAF MOLD
    if "leaf mold" in question_lower:

        return (
            "🍃 Leaf Mold\n\n"
            "Symptoms:\n"
            "Yellow spots may appear on leaves.\n\n"
            "Treatment:\n"
            "Remove infected leaves and improve ventilation.\n\n"
            "Prevention:\n"
            "Reduce humidity and avoid excessive watering."
        )


    # SPIDER MITE
    if "spider mite" in question_lower:

        return (
            "🕷️ Spider Mite\n\n"
            "Symptoms:\n"
            "Tiny yellow spots, leaf drying and webbing can occur.\n\n"
            "Treatment:\n"
            "Monitor plants and use suitable control methods.\n\n"
            "Prevention:\n"
            "Check leaves regularly."
        )
        # YELLOW LEAVES
    if (
        "yellow leaves" in question_lower
        or "yellow leaf" in question_lower
        or "leaves are yellow" in question_lower
    ):

        return (
            "🍃 Yellow Leaves\n\n"
            "Possible causes:\n"
            "• Overwatering\n"
            "• Nutrient deficiency\n"
            "• Poor drainage\n"
            "• Pests\n"
            "• Plant disease\n\n"
            "Check soil moisture and inspect leaves for "
            "spots or insects."
        )


    # WATERING
    if (
        "water" in question_lower
        or "watering" in question_lower
        or "पाणी" in question_lower
    ):

        return (
            "💧 Watering Advice\n\n"
            "Water according to soil moisture, crop type "
            "and weather conditions.\n\n"
            "Avoid overwatering and avoid keeping the soil "
            "completely dry for long periods."
        )


    # FERTILIZER
    if (
        "fertilizer" in question_lower
        or "fertiliser" in question_lower
        or "manure" in question_lower
        or "nutrient" in question_lower
        or "खत" in question_lower
    ):

        return (
            "🌱 Fertilizer Advice\n\n"
            "Fertilizer should be selected according to "
            "the crop, soil condition and growth stage.\n\n"
            "Avoid excessive fertilizer.\n\n"
            "A soil test is useful for deciding the "
            "correct nutrient requirement."
        )


    # PEST
    if (
        "pest" in question_lower
        or "insect" in question_lower
        or "bug" in question_lower
        or "keed" in question_lower
        or "कीड" in question_lower
    ):

        return (
            "🐛 Pest Management Advice\n\n"
            "First identify the pest correctly.\n\n"
            "Check the underside of leaves, new shoots "
            "and stems.\n\n"
            "Use crop-appropriate pest management methods."
        )


    # PLANT CARE
    if (
        "healthy" in question_lower
        or "plant care" in question_lower
        or "care for plant" in question_lower
    ):

        return (
            "🌱 Healthy Plant Care\n\n"
            "Provide:\n"
            "• Adequate sunlight\n"
            "• Proper watering\n"
            "• Balanced nutrients\n"
            "• Good air circulation\n"
            "• Regular pest and disease monitoring"
        )


    # DISEASE
    if (
        "disease" in question_lower
        or "रोग" in question_lower
    ):

        return (
            "🌿 Plant Disease Advice\n\n"
            "You can upload a plant leaf image on "
            "the Disease Detection page.\n\n"
            "The AI model can analyze supported tomato "
            "leaf disease classes.\n\n"
            "After detection, click "
            "'Ask AI About This Disease'."
        )


    # DEFAULT RESPONSE
    return (
        "🤖 I am your Smart Farming Assistant.\n\n"
        "You can ask me about:\n"
        "🌦️ Weather and climate\n"
        "🌡️ Temperature\n"
        "💧 Watering\n"
        "🌱 Fertilizer\n"
        "🐛 Pests and insects\n"
        "🦠 Plant diseases\n"
        "🍃 Yellow leaves\n"
        "🌿 Plant care\n\n"
        "For live weather, mention the city name."
    )


# =========================================================
# DISEASE DETECTION
# =========================================================

@app.route(
    "/disease-detection",
    methods=["GET", "POST"]
)
def disease_detection():

    if request.method == "POST":

        file = request.files.get("image")

        if not file:

            return render_template(
                "disease_detection.html"
            )


        try:

            # OPEN IMAGE
            image = Image.open(
                file
            ).convert("RGB")


            # RESIZE IMAGE
            image = image.resize(
                (128, 128)
            )


            # CONVERT TO NUMPY
            image_array = np.array(
                image
            )


            # NORMALIZE
            image_array = (
                image_array / 255.0
            )


            # ADD BATCH DIMENSION
            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # MODEL PREDICTION
            print("DEBUG: Starting disease prediction", flush=True)
            start_time = time.time()
            predictions = model(
                image_array,
                training=False
            ).numpy()
            print("DEBUG: Disease prediction finished", flush=True)
            print("Prediction time: ",time.time()-start_time)


            # GET PREDICTED CLASS
            predicted_index = np.argmax(
                predictions[0]
            )


            disease = class_names[
                predicted_index
            ]


            # CONFIDENCE
            confidence = float(
                predictions[0][predicted_index]
                * 100
            )

            confidence = round(
                confidence,
                2
            )

            # SAVE DETECTION HISTORY
            save_detection(
                file.filename,
                disease,
                confidence
            )


            # GET DISEASE INFORMATION
            info = disease_info.get(
                disease,
                {
                    "symptoms":
                        "Information not available.",

                    "treatment":
                        "Information not available.",

                    "prevention":
                        "Information not available."
                }
            )


            # SEND RESULT TO HTML
            return render_template(

                "disease_detection.html",

                disease=disease,

                confidence=confidence,

                symptoms=info["symptoms"],

                treatment=info["treatment"],

                prevention=info["prevention"]
            )


        except Exception as e:

            print(
                "Prediction Error:",
                e
            )

            return render_template(

                "disease_detection.html",

                disease="Error",

                confidence=0,

                symptoms=
                    "Unable to process the image.",

                treatment=
                    "Please upload a valid plant leaf image.",

                prevention=
                    "Try another clear image."
            )


    return render_template(
        "disease_detection.html"
    )

# =====================================
# DETECTION HISTORY PAGE
# =====================================

@app.route("/detection-history")
def detection_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    history = conn.execute("""
        SELECT id, image_name, disease, confidence, detected_at
        FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "detection_history.html",
        history=history
    )
# =====================================
# FARMER DASHBOARD
# =====================================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/chat-history")
def chat_history():
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT question, answer, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (session["user_id"],)
        )

        chats = cursor.fetchall()

        return render_template(
            "chat_history.html",
            chats=chats
        )

    except mysql.connector.Error:
        app.logger.exception("Could not load chat history")
        return "Could not load chat history.", 500

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()

@app.route("/profile")
def profile():
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT name, email, mobile, created_at
            FROM users
            WHERE id = %s
            """,
            (session["user_id"],)
        )

        user = cursor.fetchone()

        if user is None:
            session.clear()
            return redirect(url_for("login"))

        return render_template("profile.html", user=user)

    except mysql.connector.Error:
        app.logger.exception("Could not load user profile")
        return "Could not load profile. Check the Flask terminal.", 500

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )