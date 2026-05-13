from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from fastapi.staticfiles import StaticFiles
import joblib
from pydantic import BaseModel
from typing import List
import sqlite3
import pandas as pd
import os
from ASR        import transcribe
from NLU        import process_query
from RAG        import route_query
from Predict  import predict_next_race
from Respond  import format_response
from pydantic import BaseModel



app = FastAPI(title="Roma F1 Voice Assistant")

# Allow browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(BASE_DIR, "front")
DB_PATH=r"C:\projects\2nd_Semester_Project\data&models\Roma_f1_DB.db"
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")
@app.get("/")
def health_check():
    return {"status": "Roma is running "}

class DirectRequest(BaseModel):
    intent: str
    entities: dict = {}
import joblib
from pydantic import BaseModel
from typing import List

# --- 1. Analytics Compare API ---
class CompareRequest(BaseModel):
    driver_ids: List[str]
    start_year: int
    end_year: int
    metric: str

@app.post("/api/compare")
async def api_compare(req: CompareRequest):
    if not req.driver_ids:
        return {"labels": [], "datasets": []}
        
    conn = sqlite3.connect(DB_PATH)
    labels = list(range(req.start_year, req.end_year + 1)) # Pure Python ints
    datasets = []
    colors = ["#E8192C", "#00B4C8", "#F5A623", "#9B59B6"]
    
    # Bridge the gap between OpenF1 (2026) IDs and Ergast (Historical) IDs
    historical_map = {
        "lewis_hamilton": "hamilton",
        "charles_leclerc": "leclerc",
        "lando_norris": "norris",
        "oscar_piastri": "piastri",
        "carlos_sainz": "sainz",
        "george_russell": "russell",
        "fernando_alonso": "alonso",
        "sergio_perez": "perez"
    }

    for i, driver_id in enumerate(req.driver_ids):
        # Query both the full ID and the historical last-name ID together
        d_ids = [driver_id]
        if driver_id in historical_map:
            d_ids.append(historical_map[driver_id])
            
        placeholders_d = ",".join(["?" for _ in d_ids])
        placeholders_s = ",".join(["?" for _ in labels])
        
        # Using MAX(driver_name) to ensure we always get a valid string for the chart legend
        if req.metric == "points":
            query = f"SELECT season, SUM(points) as value, MAX(driver_name) as d_name FROM race_results WHERE driver_id IN ({placeholders_d}) AND season IN ({placeholders_s}) GROUP BY season ORDER BY season"
        elif req.metric == "wins":
            query = f"SELECT season, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as value, MAX(driver_name) as d_name FROM race_results WHERE driver_id IN ({placeholders_d}) AND season IN ({placeholders_s}) GROUP BY season ORDER BY season"
        elif req.metric == "podiums":
            query = f"SELECT season, SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as value, MAX(driver_name) as d_name FROM race_results WHERE driver_id IN ({placeholders_d}) AND season IN ({placeholders_s}) GROUP BY season ORDER BY season"
        elif req.metric == "avg_finish":
            query = f"SELECT season, AVG(finish_position) as value, MAX(driver_name) as d_name FROM race_results WHERE driver_id IN ({placeholders_d}) AND season IN ({placeholders_s}) AND finish_position IS NOT NULL GROUP BY season ORDER BY season"
        
        df = pd.read_sql(query, conn, params=d_ids + labels)
        
        if not df.empty:
            # Clean up the name for the Chart.js legend
            display_name = str(df['d_name'].iloc[-1]).title()
            
            data = []
            for s in labels:
                row = df[df['season'] == s]
                if not row.empty:
                    data.append(float(row['value'].iloc[0])) # Pure Python float
                else:
                    data.append(None)
                    
            datasets.append({
                "label": display_name,
                "data": data,
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)],
                "tension": 0.3,
                "spanGaps": True # Connects the line even if a driver missed a year
            })
            
    conn.close()
    return {"labels": labels, "datasets": datasets}
# --- 2. Virtual Race Simulator API ---
class SimulateRequest(BaseModel):
    grid_position: int
    front_row: int
    top5_grid: int
    champ_position: int
    champ_points: int
    champ_wins: int
    team_position: int
    team_points: int
    avg_finish_last5: float
    win_rate_last5: float
    podium_rate_last5: float
    dnf_rate_last5: float
    circuit_win_rate: float
    season_stage: float
    is_wet_race: int

@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    winner_model = joblib.load(r"C:\projects\2nd_Semester_Project\data&models\roma_best_model.pkl")
    FEATURES = joblib.load(r"C:\projects\2nd_Semester_Project\data&models\roma_features.pkl")
    
    base_row = req.dict()
    # Derived features logic
    base_row["grid_vs_champ"] = req.grid_position - req.champ_position
    base_row["avg_points_last5"] = (req.champ_points / max(1, round(req.season_stage * 24))) * req.win_rate_last5 * 25
    base_row["avg_grid_last5"] = req.grid_position
    base_row["driver_recent_points"] = (req.champ_points / max(1, round(req.season_stage * 24))) * 3
    base_row["team_avg_points_last5"] = (req.team_points / max(1, round(req.season_stage * 24))) * 3
    base_row["driver_circut_avg"] = req.avg_finish_last5
    
    # Fill remaining OHE features with 0
    row = {f: base_row.get(f, 0) for f in FEATURES}
    df = pd.DataFrame([row])
    
    prob = winner_model.predict_proba(df)[0][1]
    
    # 🚨 FIX: Explicitly cast the NumPy float to a native Python float
    win_prob = float(round(prob * 100, 1))
    
    return {"win_probability": win_prob}
@app.post("/roma")
async def roma_endpoint(audio: UploadFile = File(...)):
    start = time.time()

    # ── Step 1: ASR ───────────────────────────────────────────────
    audio_bytes = await audio.read()
    arabic_text = transcribe(audio_bytes)

    if not arabic_text:
        return JSONResponse({
            "success":  False,
            "response": "ما سمعتك، حاول مرة ثانية.",
            "debug":    {"step": "asr", "text": ""}
        })

    # ── Step 2: NLU ───────────────────────────────────────────────
    nlu_result = process_query(arabic_text)
    intent     = nlu_result["intent"]
    entities   = nlu_result["entities"]

    # ── Step 3: Route to RAG or ML ───────────────────────────────
    if intent in ("predict_next_race", "predict_next_n_races"):
        n_races_val = entities.get("Number", 1)
        if isinstance(n_races_val, list):
            n_races_val = n_races_val[0]
        n_races = int(n_races_val)
        intent  = "predict_next_race" if n_races == 1 else "predict_next_n_races"
        data    = predict_next_race(n_races=n_races)
    else:
        rag_result = route_query(intent, entities)
        data       = rag_result.get("data")

    # ── Step 4: Format Arabic Response ───────────────────────────
    arabic_response = format_response(intent, data, entities)
    elapsed = round(time.time() - start, 2)

    return JSONResponse({
        "success":        True,
        "transcription":  arabic_text,
        "intent":         intent,
        "entities":       entities,
        "response":       arabic_response,
        "latency_sec":    elapsed,
    })

@app.post("/transcribe")
async def transcribe_only(audio: UploadFile = File(...)):
    """Standalone transcription endpoint — useful for testing ASR."""
    audio_bytes = await audio.read()
    text = transcribe(audio_bytes)
    return {"text": text}

@app.post("/nlu")
async def nlu_only(text: str):
    """Standalone NLU endpoint — useful for testing intent classification."""
    result = process_query(text)
    return result
