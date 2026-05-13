import sqlite3

def format_response(intent: str, data, entities):
    """
    Converts structured data into natural Levantine Arabic text.
    No LLM — pure template-based generation.
    """

    if intent == "get_driver_standings":
        if not data:
            return "ما لقيت بيانات لترتيب السائقين."
        top5 = data[:5]
        lines = [" ترتيب بطولة السائقين الان:\n"]
        for d in top5:
            lines.append(
                f"{d['position']}. {d['driver_name']} "
                f"({d['team_id']}) — {d['points']} نقطة"
            )
        return "\n".join(lines)

    elif intent == "get_constructor_standings":
        if not data:
            return "ما لقيت بيانات لترتيب الفرق."
        lines = [" ترتيب بطولة الصانعين الان:\n"]
        for d in data[:5]:
            lines.append(
                f"{d['position']}. {d['team_name']} — {d['points']} نقطة"
            )
        return "\n".join(lines)

    elif intent == "get_race_calendar":
        if not data:
            return "ما في سباقات قادمة متاحة الان."
        lines = [" السباقات القادمة:\n"]
        for r in data[:5]:
            lines.append(f"🏁 {r['circuit']} ({r['country']}) — {r['date']}")
        return "\n".join(lines)

    elif intent == "get_race_result":
        if not data:
            return "ما لقيت نتائج للسباق المطلوب."
        lines = [" نتيجة السباق:\n"]
        for d in data[:5]:
            lines.append(
                f"{d['finish_position']}. {d['driver_name']} "
                f"({d['team_name']}) — {d['points']} نقطة"
            )
        return "\n".join(lines)

    elif intent == "get_driver_info":
        if not data:
            return "ما لقيت معلومات عن هالسائق."
        driver_name = data[0].get("driver_name", "السائق")
        wins    = sum(1 for r in data if r.get("finish_position") == 1)
        podiums = sum(1 for r in data if r.get("finish_position", 99) <= 3)
        points  = sum(r.get("points", 0) for r in data)
        return (
            f" إحصائيات {driver_name} (آخر {len(data)} سباق):\n"
            f" انتصارات: {wins}\n"
            f" بوديومات: {podiums}\n"
            f" نقاط: {points}"
        )

    elif intent == "compare_drivers":
        if not data or len(data) < 2:
            return "محتاج اسم سائقين عشان أقارن بينهم."
        d1, d2 = data[0], data[1]
        winner = d1 if d1["points"] > d2["points"] else d2
        return (
            f" مقارنة السائقين:\n\n"
            f"🔵 {d1['driver_name']}: {d1['points']} نقطة — "
            f"المركز {d1['position']}\n"
            f"🔴 {d2['driver_name']}: {d2['points']} نقطة — "
            f"المركز {d2['position']}\n\n"
            f"✅ {winner['driver_name']} أحسن هالموسم بالنقاط!"
        )
    
    elif intent == "predict_next_race":
        if not data:
            return "ما قدرت أجيب توقع الان، جرب بعد شوي."
        
        # 🚨 درع الحماية: إذا رجعت البيانات كقائمة داخل قائمة بسبب خطأ بالصوت، نأخذ أول سباق فقط 🚨
        if isinstance(data[0], list):
            race_data = data[0]
        else:
            race_data = data
            
        circuit = race_data[0].get("circuit_name", "السباق القادم")
        response = f"🔮 توقعي لسباق {circuit}:\n\n"
        
        seen_drivers = set()
        unique_drivers = []
        
        for driver in race_data:
            raw_name = driver['driver_id'].replace('_', ' ').title()
            
            # توحيد اسم أنتونيللي
            if "Antonelli" in raw_name:
                clean_name = "Kimi Antonelli"
            else:
                clean_name = raw_name
                
            if clean_name not in seen_drivers:
                seen_drivers.add(clean_name)
                driver['display_name'] = clean_name
                unique_drivers.append(driver)
                
            if len(unique_drivers) == 5:
                break
                
        for i, driver in enumerate(unique_drivers):
            name = driver['display_name']
            prob = driver.get('win_probability_%', 0)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            response += f"{medal} {name} — {prob:.1f}%\n"
        
        return response


    elif intent == "predict_next_n_races":
        if not data:
            return "ما قدرت أجيب توقعات الان."
        
        lines = [f"🔮 توقعاتي للـ {len(data)} سباقات القادمة:\n"]
        
        for race in data:
            top = race[0] if isinstance(race, list) else race
            raw_name = top.get('driver_id', '').replace('_', ' ').title()
            name = "Kimi Antonelli" if "Antonelli" in raw_name else raw_name
            prob    = top.get('win_probability_%', 0)
            circuit = top.get('circuit_name', f"سباق {top.get('race_number', '')}")
            
            lines.append(f"🏁 {circuit}: 🥇 {name} ({prob:.1f}%)")
        
        return "\n".join(lines)