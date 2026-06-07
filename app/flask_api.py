"""Flask REST API — GET endpoints over the London AQI forecasting service."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from src import config
from src.feature_pipeline.feature_store import fetch_features
from src.inference_pipeline.predict import (
    daily_summary,
    predict_current,
    predict_next_72h,
)
from src.training_pipeline.model_registry import best_model_name
from src.utils.alerts import check_alerts, classify_aqi

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

CITY = config.DEFAULT_CITY


def _err(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "aqi-predictor", "city": CITY})


@app.get("/current")
def current():
    cur = predict_current(CITY)
    cur.update(classify_aqi(cur["aqi"]))
    return jsonify(cur)


@app.get("/predict")
def predict():
    model = request.args.get("model")
    try:
        fc = predict_next_72h(CITY, model)
    except Exception as e:  # noqa: BLE001
        return _err(str(e), 503)
    return jsonify({
        "city": CITY,
        "model_used": fc["model_used"].iloc[0],
        "forecast": fc.assign(timestamp=fc["timestamp"].astype(str)).to_dict("records"),
        "daily_summary": daily_summary(fc).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    })


@app.get("/history")
def history():
    limit = int(request.args.get("limit", 72))
    df = fetch_features(CITY, limit=limit)
    if df is None or df.empty:
        return jsonify({"city": CITY, "history": []})
    cols = [c for c in ["timestamp", "aqi", "pm25", "pm10"] if c in df.columns]
    df = df[cols].assign(timestamp=df["timestamp"].astype(str))
    return jsonify({"city": CITY, "history": df.to_dict("records")})


@app.get("/alerts")
def alerts():
    try:
        fc = predict_next_72h(CITY)
    except Exception as e:  # noqa: BLE001
        return _err(str(e), 503)
    al = check_alerts(fc)
    return jsonify({"city": CITY, "alert_count": len(al), "alerts": al})


@app.get("/models")
def models():
    return jsonify({"best_model": best_model_name(CITY)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
