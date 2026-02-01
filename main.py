import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="API Deteksi Phishing - Skripsi")

# --- 1. KONFIGURASI GLOBAL ---
# Folder tempat model disimpan (harus sama dengan train_model.py)
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_phishing.json")
EXPLAINER_FILE = os.path.join(MODEL_DIR, "shap_explainer.pkl")

# Variabel Global untuk menyimpan model di memori
model = None
explainer = None

# URUTAN FITUR (PENTING: Harus SAMA PERSIS dengan train_model.py)
feature_names = [
    "n_urls", 
    "n_domains", 
    "n_attachments", 
    "has_password_word", 
    "min_domain_age_days", 
    "count_suspicious_tld", 
    "is_spf_fail"
]

# --- 2. LOAD MODEL SAAT STARTUP ---
@app.on_event("startup")
async def startup_event():
    global model, explainer
    print("⏳ Sedang memuat model AI...")
    
    # Cek apakah file model ada
    if not os.path.exists(MODEL_FILE) or not os.path.exists(EXPLAINER_FILE):
        print(f"❌ ERROR: File model tidak ditemukan di {MODEL_DIR}!")
        print("⚠️  Tolong jalankan 'python train_model.py' terlebih dahulu.")
        return

    try:
        # Load XGBoost
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        
        # Load SHAP Explainer
        with open(EXPLAINER_FILE, "rb") as f:
            explainer = pickle.load(f)
            
        print("✅ SIAP! Model berhasil dimuat dan API berjalan.")
        print(f"📂 Menggunakan model dari: {MODEL_DIR}")
        
    except Exception as e:
        print(f"❌ Error saat loading model: {e}")

# --- 3. INPUT SCHEMA (Format JSON dari n8n) ---
class EmailInput(BaseModel):
    # Field ini harus sesuai dengan output JSON node n8n kamu
    urls: List[str] = []
    domains: List[str] = []
    hasPasswordWord: int = 0  # 1 jika ada kata password, 0 jika tidak
    n_attachments: int = 0
    domain_ages: List[int] = [] # List umur domain dalam hari
    spf_status: str = "pass"    # pass, fail, softfail, neutral, none

# --- 4. PREPROCESSING (Ubah JSON n8n jadi DataFrame) ---
def preprocess_input(data: EmailInput) -> pd.DataFrame:
    """
    Mengubah data mentah dari n8n menjadi format angka yang dimengerti model.
    """
    
    # 1. Hitung n_urls & n_domains
    n_urls = len(data.urls)
    n_domains = len(data.domains)
    
    # 2. Cari domain termuda (Paling bahaya)
    # Jika n8n gagal dapet umur (list kosong), kita anggap aman (misal 3 tahun/1095 hari)
    # supaya tidak error/NaN
    if data.domain_ages:
        min_domain_age = min(data.domain_ages)
    else:
        min_domain_age = 1095 
        
    # 3. Hitung TLD mencurigakan (.xyz, .top, dll)
    # List TLD ini bisa kamu tambah sendiri nanti
    suspicious_tlds = ['.xyz', '.top', '.club', '.gq', '.info', '.work', '.biz']
    count_suspicious = sum(1 for d in data.domains if any(d.endswith(tld) for tld in suspicious_tlds))
    
    # 4. Cek SPF Failure
    # Jika status bukan 'pass', kita anggap fail (1)
    is_spf_fail = 1 if data.spf_status.lower() != "pass" else 0

    # Susun ke DataFrame sesuai urutan feature_names
    features = pd.DataFrame([{
        "n_urls": n_urls,
        "n_domains": n_domains,
        "n_attachments": data.n_attachments,
        "has_password_word": data.hasPasswordWord,
        "min_domain_age_days": min_domain_age,
        "count_suspicious_tld": count_suspicious,
        "is_spf_fail": is_spf_fail
    }])
    
    return features

# --- 5. ENDPOINT UTAMA (/predict) ---
@app.post("/predict")
async def predict_email(input_data: EmailInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model belum siap/gagal dimuat.")

    # A. Preprocessing
    X_input = preprocess_input(input_data)
    
    # B. Prediksi (Probabilitas Phishing)
    # model.predict_proba return [[prob_aman, prob_phishing]]
    prob_phishing = float(model.predict_proba(X_input)[0][1])
    
    # Threshold 0.5 (Bisa kamu naikkan/turunkan sesuai kebutuhan skripsi)
    is_phishing = prob_phishing > 0.5
    label = "PHISHING" if is_phishing else "LEGITIMATE"

    # C. Hitung SHAP Values (Penjelasan)
    try:
        shap_values = explainer.shap_values(X_input)
        
        # Handle format array shap yang kadang beda versi library
        if isinstance(shap_values, list):
            # Binary classification biasanya return list [shap_class_0, shap_class_1]
            vals = shap_values[1][0] 
        elif len(shap_values.shape) > 1:
             vals = shap_values[0] # Jika bentuknya matriks (1, n_features)
        else:
             vals = shap_values # Jika bentuknya array 1D
             
    except Exception as e:
        print(f"⚠️ Warning SHAP calculation: {e}")
        # Fallback nilai 0 semua jika error, biar API gak mati
        vals = [0] * len(feature_names)

    # D. Format Penjelasan untuk LLM
    feature_impacts = []
    for i, name in enumerate(feature_names):
        val = float(X_input.iloc[0][name]) # Nilai asli fiturnya (misal: age=2)
        impact = float(vals[i])            # Nilai impact SHAP (misal: +1.2)
        
        feature_impacts.append({
            "fitur": name,
            "nilai_input": val,
            "kontribusi_bahaya": round(impact, 4)
        })
    
    # Urutkan dari yang paling berpengaruh (absolute value terbesar)
    feature_impacts.sort(key=lambda x: abs(x["kontribusi_bahaya"]), reverse=True)
    
    # Ambil Top 3 Penyebab
    top_reasons = feature_impacts[:3]

    # E. Output JSON
    return {
        "label": label,
        "score_bahaya": round(prob_phishing, 4), # 0.0 sampai 1.0
        "analisis_shap": top_reasons,
        "pesan_sistem": "Kirim JSON ini ke LLM untuk dibuatkan narasi."
    }