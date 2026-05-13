# Roma-F1-AI-Assistant
Roma is an intelligent, end-to-end Arabic voice assistant tailored for the Formula 1 universe. Specifically designed to understand the Levantine dialect (Jordanian and Palestinian), Roma provides real-time race data, historical analytics, and Machine Learning-powered race predictions. 

Built as an academic and engineering showcase, Roma operates with **zero budget** and **no external LLM wrappers** (like ChatGPT). Every layer is locally hosted, fully explainable, and deterministic.

---------------------------------------------------------
features
1.  **Voice Mode (المساعد الصوتي):** Speaks and understands Levantine Arabic using a custom fine-tuned Whisper model.
2.  **Driver & Team Analytics (تحليل الأداء):** Interactive, glassmorphism telemetry dashboard comparing driver and team performance from 2010 to 2026.
3.  **Race Prediction (توقع السباقات):** Predicts the top 5 winners for upcoming races using a machine learning model.
4.  **Virtual Race Simulator (محاكي السباق):** A highly interactive UI that allows users to tweak race conditions (Grid position, weather, recent form) and instantly see the XGBoost win probability.
---------------------------------------------------------
Browser Mic (Custom HTML/Tailwind SPA)
    ↓
1️⃣ ASR Layer: HuggingFace Whisper (Fine-tuned on Levantine Arabic, 37% WER)
    ↓
2️⃣ NLU Layer: TF-IDF + LinearSVC (99% intent accuracy) + RapidFuzz Entity Extraction
    ↓
3️⃣ Router: Intent-based decision engine
    ↓
4️⃣ RAG System: SQLite (Historical Data) + OpenF1 API (Live 2026 Data)
    OR
4️⃣ ML Engine: XGBoost Classifier (PSO-optimized, exponential decay-weighted)
    ↓
5️⃣ Response Layer: Arabic Template Generator -> FastAPI -> Frontend

Run the code 
open the front.html file then run this command 
uvicorn main:app --reload --port 8000
