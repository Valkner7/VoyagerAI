import os
import re
import json
import random
from collections import OrderedDict
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION & RECOVERY SETTINGS
# =========================================================
load_dotenv()
my_secret_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=my_secret_key)

app = FastAPI(title="Voyager.AI Production Engine")

# Full CORS layer for secure communication node validation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LIGHTWEIGHT STATE CONTEXT & CACHE STORE
# =========================================================
ITINERARY_CACHE = OrderedDict()
MAX_CACHE_SIZE = 15

class ItineraryRequest(BaseModel):
    season: str
    focus: str
    start_time: str
    variant: int = 0

# Core dataset registries used for structural data fallback
DESTINATIONS = [
    {"title": "Old City Spice Market", "category": "shopping", "icon": "🌶️", "time": "07:00 AM - 10:00 AM", "desc": "Specialty market for local spices. Best visited early.", "coords": [28.6550, 77.2243]},
    {"title": "Silk Weavers District", "category": "shopping", "icon": "👗", "time": "11:00 AM - 04:00 PM", "desc": "Renowned market for hand-woven silks.", "coords": [28.6448, 77.2167]},
    {"title": "Grand Royal Palace", "category": "heritage", "icon": "🏰", "time": "09:00 AM - 01:00 PM", "desc": "Historical center. Guided services available.", "coords": [28.6562, 77.2410]},
    {"title": "Lumina Boutique Stays", "category": "heritage", "icon": "🏨", "time": "24/7", "desc": "Luxury stays. Show app for free welcome drink.", "sponsored": True, "coords": [28.6500, 77.2300]},
    {"title": "Ancient Stepwell", "category": "heritage", "icon": "🏛️", "time": "04:00 PM - 06:00 PM", "desc": "Beautiful architectural heritage site.", "coords": [28.6266, 77.2255]},
    {"title": "Street Food Alley", "category": "food", "icon": "🍢", "time": "06:00 PM - 11:00 PM", "desc": "Authentic local culture through street delicacies.", "coords": [28.6505, 77.2305]},
    {"title": "Traditional Thali House", "category": "food", "icon": "🍛", "time": "12:30 PM - 03:00 PM", "desc": "Sit-down cultural food experience.", "coords": [28.6328, 77.2197]},
    {"title": "Global Cafe Co.", "category": "food", "icon": "☕", "time": "08:00 AM - 10:00 PM", "desc": "High-speed Wi-Fi and artisan coffee.", "sponsored": True, "coords": [28.6304, 77.2177]}
]

DELHI_COORDS_REGISTRY = {
    "chandni": [28.6505, 77.2303],
    "spice": [28.6550, 77.2243],
    "red fort": [28.6562, 77.2410],
    "jama": [28.6507, 77.2334],
    "connaught": [28.6304, 77.2177],
    "india gate": [28.6129, 77.2295],
    "humayun": [28.5933, 77.2507],
    "lodhi": [28.5931, 77.2197],
    "qutub": [28.5245, 77.1855],
    "lotus": [28.5535, 77.2588],
    "akshardham": [28.6127, 77.2773],
    "market": [28.6448, 77.2167]
}

# =========================================================
# SYSTEM UTILITY UTILITIES & ALGORITHMIC ENGINES
# =========================================================
def get_live_weather_constraints():
    """
    Fetches real-time climate telemetry variables for New Delhi coordinates
    and outputs strict optimization parameters for prompt engineering injections.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current=temperature_2m,rain"
        response = urllib.request.urlopen(url, timeout=4)
        data = json.loads(response.read().decode())
        
        current = data.get("current", {})
        temp = current.get("temperature_2m", 25)
        rain = current.get("rain", 0)
        
        if temp > 38:
            return f"CRITICAL NOTICE: Current New Delhi temperature is dangerously high ({temp}°C). To protect the traveler, strictly omit all open-air markets, outdoor walking tracks, and exposed historical ruins. Substitute them exclusively with indoor, air-conditioned environments, museums, sheltered art centers, or premium cafes."
        elif rain > 0:
            return f"CRITICAL NOTICE: Active precipitation/rainfall detected in New Delhi. Completely avoid outdoor gardens, street markets, or unroofed structures. Force all 5 itinerary steps to leverage indoor, covered, or subterranean attractions like shopping arcades, stepwells, or indoor exhibitions."
        
        return f"Current New Delhi weather is clear and moderate ({temp}°C). Generate standard highly accessible routes."
    except Exception as e:
        print(f"Weather abstraction layer fallback activated: {e}")
        return "Weather parameters are currently standard. Optimize for common seasonal guidelines."


def inject_coordinates(itinerary_list):
    """Parses text descriptions and appends real geospatial coordinates sequentially."""
    last_known_coord = [28.6200, 77.2300] 
    
    for step in itinerary_list:
        title_lower = step.get("title", "").lower()
        desc_lower = step.get("desc", "").lower() or step.get("description", "").lower()
        matched = False
        
        for key, coords in DELHI_COORDS_REGISTRY.items():
            if key in title_lower or key in desc_lower:
                step["coords"] = coords
                last_known_coord = coords
                matched = True
                break
                
        if not matched:
            step["coords"] = [
                last_known_coord[0] + random.uniform(-0.015, 0.015),
                last_known_coord[1] + random.uniform(-0.015, 0.015)
            ]
    return itinerary_list

# =========================================================
# CONTROLLER ENDPOINTS LOGIC LAYER
# =========================================================
@app.get("/")
async def root_status():
    return {"status": "online", "engine": "Voyager.AI Core"}


@app.get("/api/rates")
async def get_transport_rates():
    return {
        "bike": {"base": 0.5, "market": 0.8},
        "auto": {"base": 1.2, "market": 1.8},
        "car": {"base": 2.5, "market": 3.5}
    }


@app.get("/api/destinations")
async def get_destinations(category: str = "all"):
    if category == "all":
        return DESTINATIONS
    return [place for place in DESTINATIONS if place.get("category") == category]


@app.post("/api/generate-itinerary")
async def generate_itinerary(req: ItineraryRequest):
    # Enforce performance tracking via state cache parameters
    cache_key = (req.season.lower(), req.focus.lower(), req.start_time, req.variant)
    if cache_key in ITINERARY_CACHE:
        print(f"⚡ Cache Hit! Serving Itinerary variant {req.variant} instantly.")
        return ITINERARY_CACHE[cache_key]
        
    print(f"🤖 Cache Miss. Requesting Variant {req.variant} from Gemini with weather validation...")
    
    try:
# Fetch live meteorological data
        weather_override = get_live_weather_constraints()
        
        # Initialize the model correctly using the stable 2.5-flash engine
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # The entire prompt must live completely INSIDE these triple quotes
        system_instruction = f"""
        You are Voyager.AI, an expert travel optimization system for Delhi, India.
        
        LIVE ENVIRONMENT VARIABLES:
        {weather_override}
        
        Create a detailed, 5-step single-day itinerary for New Delhi, India during the {req.season} season focusing on {req.focus}.
        The tour must start exactly at {req.start_time}.
        This is variant request #{req.variant}, so make the selection completely distinct from standard routes.
        
        CRITICAL OUTPUT ARCHITECTURE RULES:
        1. Provide exactly 5 sequential chronological itinerary steps.
        2. Format your response STRICTLY as a valid JSON array matching this exact layout blueprint, with no markdown backticks, wrap-around text, or prose outside the array structure:
        [
          {{
            "timeStr": "09:00 AM",
            "title": "Landmark Name",
            "desc": "A brief 1-sentence description.",
            "icon": "🧭"
          }}
        ]
        """
        
        # Execute the request to Gemini
        response = model.generate_content(
    system_instruction,
    generation_config={
        "temperature":0.7,
        "response_mime_type":"application/json"
    }
)
        
        raw_text = response.text.strip()
        
        # Regex execution mapping blocks out unformatted contextual padding lines
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        clean_json_str = match.group(0) if match else raw_text
        
        raw_data = json.loads(clean_json_str)
        
        # Structure normalization check
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, list):
                    raw_data = value
                    break
                    
        if not isinstance(raw_data, list):
            raise ValueError("AI did not return a valid list structure.")
            
        processed_data = inject_coordinates(raw_data)
        
        # Maintain local server cache state bounds
        if len(ITINERARY_CACHE) >= MAX_CACHE_SIZE:
            ITINERARY_CACHE.popitem(last=False)
        ITINERARY_CACHE[cache_key] = processed_data
        
        return processed_data

    except Exception as e:
        print(f"Core execution block failure handler: {e}")
        fallback = [
            {"timeStr": "09:00 AM", "title": "Chandni Chowk Cultural Walk", "desc": "Explore historical lanes of Old Delhi.", "icon": "🧭"},
            {"timeStr": "11:30 AM", "title": "Red Fort Landmark Visit", "desc": "Stunning Mughal architecture tour.", "icon": "🏰"},
            {"timeStr": "02:00 PM", "title": "Traditional Connaught Place Lunch", "desc": "Enjoy classic food options in Delhi's heart.", "icon": "🍛"},
            {"timeStr": "04:30 PM", "title": "Humayun's Tomb Sunset View", "desc": "Beautiful garden tomb photography session.", "icon": "📸"},
            {"timeStr": "07:00 PM", "title": "Local Street Market Exploration", "desc": "Souvenir shopping and evening snacks.", "icon": "🛍️"}
        ]
        processed_fallback = inject_coordinates(fallback)
        
        if "429" in str(e) or "quota" in str(e).lower():
            return processed_fallback
            
        raise HTTPException(status_code=500, detail=str(e))
