import sqlite3
import requests
import pandas as pd
from datetime import datetime

DB_PATH      = r"C:\projects\2nd_Semester_Project\data&models\Roma_f1_DB.db"
OPENF1_BASE  = "https://api.openf1.org/v1"

def get_live(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_cached(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()
    return rows

def handle_driver_standings():
    rows = get_cached("""
    SELECT driver_id, driver_name, team_id, points, position, wins
    FROM driver_standings
    WHERE season = (SELECT MAX(season) FROM driver_standings)
      AND round  = (SELECT MAX(round) FROM driver_standings
                    WHERE season = (SELECT MAX(season) FROM driver_standings))
    ORDER BY position
""", ())

# لو فاضية احسبها من race_results
    if not rows:
        rows = get_cached("""
        SELECT driver_id, driver_name, team_name,
               SUM(points) as points,
               RANK() OVER (ORDER BY SUM(points) DESC) as position,
               SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins
        FROM race_results
        WHERE season = (SELECT MAX(season) FROM race_results)
        GROUP BY driver_id, driver_name, team_name
        ORDER BY points DESC
    """, ())
    
    return {"intent": "get_driver_standings", "data": rows}


def handle_constructor_standings():
    rows = get_cached("""
        SELECT team_id, 
               REPLACE(team_id, '_', ' ') AS team_name, 
               points, position, wins
        FROM constructor_standings
        WHERE season = (SELECT MAX(season) FROM constructor_standings)
          AND round  = (SELECT MAX(round) FROM constructor_standings
                        WHERE season = (SELECT MAX(season) FROM constructor_standings))
        ORDER BY position
    """)
    return {"intent": "get_constructor_standings", "data": rows}

def handle_race_calendar():
    from datetime import datetime, timezone
    F1_2026_CALENDAR = [
        {"circuit": "أستراليا",        "country": "أستراليا",      "date": "2026-03-15"},
        {"circuit": "الصين",            "country": "الصين",          "date": "2026-03-22"},
        {"circuit": "اليابان",          "country": "اليابان",        "date": "2026-04-05"},
        {"circuit": "ميامي",            "country": "أمريكا",         "date": "2026-05-03"},
        {"circuit": "كندا",             "country": "كندا",           "date": "2026-05-24"},
        {"circuit": "موناكو",           "country": "موناكو",         "date": "2026-06-07"},
        {"circuit": "برشلونة كتالونيا", "country": "إسبانيا",        "date": "2026-06-14"},
        {"circuit": "النمسا",           "country": "النمسا",         "date": "2026-06-28"},
        {"circuit": "بريطانيا",         "country": "بريطانيا",       "date": "2026-07-05"},
        {"circuit": "بلجيكا",           "country": "بلجيكا",         "date": "2026-07-19"},
        {"circuit": "المجر",            "country": "المجر",          "date": "2026-07-26"},
        {"circuit": "هولندا",           "country": "هولندا",         "date": "2026-08-23"},
        {"circuit": "إيطاليا",          "country": "إيطاليا",        "date": "2026-09-06"},
        {"circuit": "إسبانيا",          "country": "إسبانيا",        "date": "2026-09-13"},
        {"circuit": "أذربيجان",         "country": "أذربيجان",       "date": "2026-09-26"},
        {"circuit": "سنغافورة",         "country": "سنغافورة",       "date": "2026-10-11"},
        {"circuit": "الولايات المتحدة", "country": "أمريكا",         "date": "2026-10-25"},
        {"circuit": "المكسيك",          "country": "المكسيك",        "date": "2026-11-01"},
        {"circuit": "البرازيل",         "country": "البرازيل",       "date": "2026-11-08"},
        {"circuit": "لاس فيغاس",        "country": "أمريكا",         "date": "2026-11-21"},
        {"circuit": "قطر",              "country": "قطر",            "date": "2026-11-29"},
        {"circuit": "أبوظبي",           "country": "الإمارات",       "date": "2026-12-06"},
    ]
    now = datetime.now(timezone.utc).date()
    upcoming = [
        r for r in F1_2026_CALENDAR
        if datetime.strptime(r["date"], "%Y-%m-%d").date() >= now
    ]
    return {"intent": "get_race_calendar", "data": upcoming[:8]}


def handle_race_result(entities):
    circuit   = entities.get("Circuits")
    circuit_filter = f"AND circuit_id = '{circuit}'" if circuit else ""
    rows = get_cached(f"""
        SELECT season, round, circuit_name, driver_id,
               driver_name, team_name, finish_position, points
        FROM race_results
        WHERE season = (SELECT MAX(season) FROM race_results)
        {circuit_filter}
        ORDER BY round DESC, finish_position ASC
        LIMIT 20
    """)
    return {"intent": "get_race_result", "data": rows}

def handle_driver_info(entities):
    drivers = entities.get("Drivers", [])
    if not drivers:
        return {"intent": "get_driver_info", "data": []}
    driver_id = drivers[0]
    rows = get_cached("""
        SELECT season, round, circuit_name, driver_name,
               team_name, finish_position, points, status
        FROM race_results
        WHERE driver_id = ?
          AND season >= 2024
        ORDER BY season DESC, round DESC
        LIMIT 15
    """, (driver_id,))
    return {"intent": "get_driver_info", "data": rows}

def handle_compare_drivers(entities):
    drivers = entities.get("Drivers", [])
    if len(drivers) < 2:
        return {"intent": "compare_drivers", "data": []}
    results = []
    for d in drivers[:2]:
        rows = get_cached("""
            SELECT driver_id, 
                   REPLACE(driver_id, '_', ' ') AS driver_name, 
                   '' AS team_name,
                   points, position, wins
            FROM driver_standings
            WHERE driver_id = ?
              AND season = (SELECT MAX(season) FROM driver_standings)
              AND round  = (SELECT MAX(round) FROM driver_standings
                            WHERE season = (SELECT MAX(season) FROM driver_standings))
        """, (d,))
        results.extend(rows)
    return {"intent": "compare_drivers", "data": results}

def route_query(intent, entities):

    if intent == "get_driver_standings":
        return handle_driver_standings()
    elif intent == "get_constructor_standings":
        return handle_constructor_standings()
    elif intent == "get_race_calendar":
        return handle_race_calendar()
    elif intent == "get_race_result":
        return handle_race_result(entities)
    elif intent == "get_driver_info":
        return handle_driver_info(entities)
    elif intent == "compare_drivers":
        return handle_compare_drivers(entities)
    elif intent in ("predict_next_race", "predict_next_n_races"):
        return {"intent": intent, "data": None, "route": "ml_model"}
    return {"intent": "unknown", "data": None}