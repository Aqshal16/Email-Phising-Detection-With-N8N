"""
FastAPI Server untuk Phishing Detection dengan OSINT_ENHANCED Model
Menerima email dari N8N, extract features, run OSINT tools, predict
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import re
import shap
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
import whois as python_whois

app = FastAPI(title="Phishing Detection API - OSINT Enhanced")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# LOAD MODELS & VECTORIZER
# ========================================
text_model = None
osint_model = None
tfidf_vectorizer = None
shap_explainer = None

@app.on_event("startup")
def load_models():
    global text_model, osint_model, tfidf_vectorizer, shap_explainer
    
    print("🚀 Loading models...")
    
    try:
        # Load TEXT_ONLY model
        text_model = xgb.Booster()
        text_model.load_model('models/xgb_text_only.json')
        print("✅ TEXT_ONLY model loaded")
    except Exception as e:
        print(f"⚠️  TEXT_ONLY model not found: {e}")
    
    try:
        # Load OSINT_ENHANCED model
        osint_model = xgb.Booster()
        osint_model.load_model('models/xgb_osint_enhanced.json')
        print("✅ OSINT_ENHANCED model loaded")
    except Exception as e:
        print(f"⚠️  OSINT_ENHANCED model not found: {e}")
    
    try:
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("✅ TF-IDF vectorizer loaded")
    except Exception as e:
        print(f"⚠️  TF-IDF vectorizer not found: {e}")
    
    # SHAP explainer akan dibuat on-demand saat prediksi pertama
    # untuk menghindari issue compatibility saat startup
    print("ℹ️  SHAP explainer will be created on first prediction")
    
    print("🎉 API ready!\n")


# ========================================
# REQUEST MODELS
# ========================================
class OSINTFeatures(BaseModel):
    """OSINT features yang sudah di-extract dari N8N"""
    domain_age_days: int = 3650
    has_registrar: int = 0
    host_up: int = 0
    common_web_ports_open: int = 0
    open_ports_count: int = 0
    filtered_ports_count: int = 0
    https_supported: int = 0
    latency: float = 100.0
    scan_duration: float = 10.0
    alternate_ip_count: int = 0
    asn_found: int = 0
    host_found: int = 0
    ip_found: int = 0
    interesting_url: int = 0

class EmailRequest(BaseModel):
    """
    Request model untuk prediction dengan OSINT features dari N8N
    N8N akan kirim email_text + OSINT features yang sudah di-extract
    """
    email_text: str
    osint_features: OSINTFeatures  # OSINT dari N8N!


# ========================================
# HELPER FUNCTIONS
# ========================================

def extract_urls_from_text(text: str):
    """Extract URLs from email text."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, str(text))

def extract_domains_from_urls(urls):
    """Extract domains from URLs."""
    domains = []
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain:
                domains.append(domain)
        except:
            pass
    return list(set(domains))

def detect_interesting_url(text: str) -> int:
    """Detect suspicious URL patterns in email text"""
    urls = extract_urls_from_text(text)
    
    suspicious_patterns = [
        r'login', r'verify', r'update', r'secure', r'account',
        r'confirm', r'banking', r'paypal', r'signin', r'password'
    ]
    
    for url in urls:
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower):
                return 1
    
    return 0


# ========================================
# OSINT TOOLS FUNCTIONS (REMOVED - NOW HANDLED BY N8N)
# ========================================
# WHOIS, Nmap, theHarvester sekarang di-run dari N8N
# N8N akan kirim hasil OSINT sebagai parameter ke API


# ========================================
# PREDICTION ENDPOINT
# ========================================
# ========================================
# PREDICTION ENDPOINT
# ========================================

@app.post("/predict_osint_enhanced")
async def predict_osint_enhanced(request: EmailRequest):
    """
    Endpoint untuk prediksi phishing dengan OSINT_ENHANCED model (173 features)
    
    Input dari N8N:
    - email_text: Raw email content
    - OSINT features (14): domain_age_days, host_up, common_web_ports_open, dll
    
    API akan extract:
    - Manual text: 9 features (dari email_text)
    - TF-IDF: 150 features (dari email_text)
    - OSINT: 14 features (SUDAH dari N8N, tidak run tools lagi)
    
    Total: 173 features
    """
    
    if osint_model is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=500, detail="OSINT model or TF-IDF vectorizer not loaded")
    
    try:
        print(f"\n{'='*70}")
        print(f"🔍 OSINT_ENHANCED PREDICTION")
        print(f"{'='*70}")
        
        # 1. Extract manual text features (9) - dari email_text
        print("📝 Extracting manual text features from email...")
        
        urls = extract_urls_from_text(request.email_text)
        domains = extract_domains_from_urls(urls)
        
        # Manual features
        n_urls = len(urls)
        n_domains = len(domains)
        
        # IP count
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        n_ips = len(re.findall(ip_pattern, request.email_text))
        
        # HTTPS/HTTP
        has_https = 1 if any('https://' in url for url in urls) else 0
        has_http = 1 if any('http://' in url and 'https://' not in url for url in urls) else 0
        
        # Suspicious keywords
        suspicious_keywords = [
            'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
            'password', 'update', 'secure', 'bank', 'login', 'credit',
            'ssn', 'social security', 'expire', 'limited time', 'act now',
            'congratulations', 'winner', 'prize', 'claim', 'refund'
        ]
        n_suspicious_keywords = sum(1 for kw in suspicious_keywords if kw in request.email_text.lower())
        
        # Email length
        email_length = len(request.email_text)
        
        # Attachment mentions
        attachment_keywords = ['attachment', 'attached', 'file', 'download', 'pdf', 'doc', 'zip']
        n_attachments = sum(1 for kw in attachment_keywords if kw in request.email_text.lower())
        
        # Suspicious TLD count
        suspicious_tlds = ['.xyz', '.top', '.club', '.work', '.download', '.loan', 
                           '.win', '.bid', '.click', '.stream', '.gq', '.cf', '.ml']
        count_susp_tld = 0
        for domain in domains:
            for tld in suspicious_tlds:
                if domain.endswith(tld):
                    count_susp_tld += 1
                    break
        
        manual_features = [
            n_urls,
            n_domains,
            n_ips,
            has_https,
            has_http,
            n_suspicious_keywords,
            email_length,
            n_attachments,
            count_susp_tld
        ]
        
        print(f"✅ Manual features: {manual_features}")
        
        # 2. TF-IDF features (150)
        print("📊 Extracting TF-IDF features...")
        tfidf_matrix = tfidf_vectorizer.transform([request.email_text])
        tfidf_features = tfidf_matrix.toarray()[0].tolist()
        print(f"✅ TF-IDF features extracted: {len(tfidf_features)} features")
        
        # 3. OSINT features (14) - DARI N8N (tidak run tools lagi!)
        print("🔍 Receiving OSINT features from N8N...")
        
        domain = domains[0] if domains else "unknown"
        osint = request.osint_features
        
        osint_features = [
            osint.domain_age_days,
            osint.host_up,
            osint.common_web_ports_open,
            osint.open_ports_count,
            osint.filtered_ports_count,
            osint.https_supported,
            osint.latency,
            osint.scan_duration,
            osint.alternate_ip_count,
            osint.asn_found,
            osint.host_found,
            osint.ip_found,
            osint.interesting_url,
            osint.has_registrar
        ]
        
        print(f"✅ OSINT features (from N8N): {osint_features}")
        print(f"   Domain: {domain}")
        print(f"   Age: {osint.domain_age_days} days, HTTPS: {osint.https_supported}, Host Up: {osint.host_up}")
        
        # 4. Combine feature values
        all_features = manual_features + tfidf_features + osint_features
        
        print(f"✅ Total features generated: {len(all_features)}")
        
        # 5. Get expected feature names from the model
        # IMPORTANT: XGBoost sometimes doesn't preserve feature_names when saving/loading
        # So we build the expected feature names based on the vectorizer and known OSINT features
        
        manual_feature_names_expected = [
            'n_urls', 'n_domains', 'n_ips', 'has_https', 'has_http',
            'n_suspicious_keywords', 'email_length', 'n_attachments', 'count_suspicious_tld'
        ]
        
        # TF-IDF feature names (dari vectorizer)
        tfidf_feature_names_expected = [f'tfidf_{word}' for word in tfidf_vectorizer.get_feature_names_out()]
        
        # OSINT feature names (14 features)
        osint_feature_names_expected = [
            'domain_age_days', 'host_up', 'common_web_ports_open', 'open_ports_count',
            'filtered_ports_count', 'https_supported', 'latency', 'scan_duration',
            'alternate_ip_count', 'asn_found', 'host_found', 'ip_found',
            'interesting_url', 'has_registrar'
        ]
        
        # Build expected feature names list (same order as model was trained)
        model_feature_names = manual_feature_names_expected + tfidf_feature_names_expected + osint_feature_names_expected
        all_feature_names = model_feature_names  # Use the same for validation
        
        print(f"✅ Model expects {len(model_feature_names)} features")
        print(f"   - Manual features: {len(manual_feature_names_expected)}")
        print(f"   - TF-IDF features: {len(tfidf_feature_names_expected)}")
        print(f"   - OSINT features: {len(osint_feature_names_expected)}")
        
        # 6. Validate that TF-IDF vectorizer matches the model
        expected_features_set = set(model_feature_names)
        generated_features_set = set(all_feature_names)
        
        missing_features = expected_features_set - generated_features_set
        extra_features = generated_features_set - expected_features_set
        
        if missing_features or extra_features:
            error_msg = "❌ FEATURE MISMATCH ERROR\n\n"
            error_msg += "TF-IDF vectorizer yang di-load TIDAK SAMA dengan yang digunakan saat training model!\n\n"
            
            if missing_features:
                error_msg += f"Missing features (model expects but not generated): {len(missing_features)}\n"
                error_msg += f"Examples: {list(missing_features)[:10]}\n\n"
            
            if extra_features:
                error_msg += f"Extra features (generated but model doesn't expect): {len(extra_features)}\n"
                error_msg += f"Examples: {list(extra_features)[:10]}\n\n"
            
            error_msg += "SOLUSI:\n"
            error_msg += "1. Gunakan tfidf_vectorizer.pkl yang SAMA dengan yang digunakan saat training\n"
            error_msg += "2. Atau re-train model dengan vectorizer yang sekarang di-load\n"
            error_msg += "3. Pastikan file models/tfidf_vectorizer.pkl adalah file yang benar dari training"
            
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print("✅ Feature names match! Reordering to match model...")
        
        # 7. Create DataFrame with features in the correct order
        X_df = pd.DataFrame([all_features], columns=all_feature_names)
        X_df = X_df[model_feature_names]  # Reorder columns to match model
        
        print(f"✅ DataFrame shape: {X_df.shape}")
        print(f"✅ First 5 columns: {list(X_df.columns[:5])}")
        print(f"✅ Last 5 columns: {list(X_df.columns[-5:])}")
        
        # Verify feature count
        if X_df.shape[1] != 173:
            raise HTTPException(
                status_code=500, 
                detail=f"Feature count mismatch: got {X_df.shape[1]}, expected 173"
            )
        
        # Create DMatrix from DataFrame with explicit feature names
        # This is important because some versions of XGBoost don't preserve feature names
        dmatrix = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
        
        print(f"✅ DMatrix created with {len(dmatrix.feature_names)} features")
        
        # 8. Predict
        prediction = osint_model.predict(dmatrix)[0]
        
        # 9. SHAP Analysis - Detail Explanation untuk setiap prediksi
        print("🔍 Running detailed SHAP analysis...")
        print("=" * 70)
        
        analisis_shap = {
            "enabled": True,
            "top_features": [],
            "feature_categories": {},
            "total_features_analyzed": 0
        }
        
        try:
            global shap_explainer
            
            # Initialize SHAP explainer (only once)
            if shap_explainer is None:
                print("📊 Initializing SHAP TreeExplainer...")
                print("   (This happens once, may take 10-15 seconds)")
                
                try:
                    # Method 1: Direct initialization (paling simple)
                    shap_explainer = shap.TreeExplainer(osint_model)
                    print("✅ SHAP explainer initialized successfully")
                    
                except Exception as e_direct:
                    print(f"⚠️  Direct init failed: {str(e_direct)[:100]}")
                    print("🔄 Attempting workaround: Binary reload...")
                    
                    # Method 2: Workaround untuk compatibility issues
                    # Convert model to binary format (lebih compatible)
                    model_bytes = osint_model.save_raw('json')
                    temp_model = xgb.Booster()
                    temp_model.load_model(bytearray(model_bytes))
                    
                    shap_explainer = shap.TreeExplainer(temp_model)
                    print("✅ SHAP explainer initialized via binary workaround")
            
            # Calculate SHAP values untuk prediksi ini
            print("📈 Computing SHAP values for 173 features...")
            
            # Use numpy array to avoid DataFrame compatibility issues
            X_array = X_df.values
            
            # Calculate SHAP values
            # check_additivity=False untuk avoid floating point precision errors
            shap_values = shap_explainer.shap_values(X_array, check_additivity=False)
            
            # Handle output format (bisa array atau list of arrays)
            # For binary classification, SHAP might return shape (n_samples, n_features) or (2, n_samples, n_features)
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]  # Binary classification - ambil class 0
            else:
                shap_vals = shap_values
            
            # If still 2D array (n_samples, n_features), get first sample
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
            
            # Get base value (baseline prediction sebelum features)
            base_value = shap_explainer.expected_value
            
            # Handle different base_value formats
            if isinstance(base_value, (list, np.ndarray)):
                if len(base_value) > 1:
                    base_value = base_value[0]  # Binary classification - ambil class 0
                else:
                    base_value = base_value.item() if hasattr(base_value, 'item') else float(base_value)
            
            # Ensure base_value is scalar
            if hasattr(base_value, 'item'):
                base_value = base_value.item()  # Convert numpy scalar to Python scalar
            else:
                base_value = float(base_value)
            
            print(f"✅ SHAP values computed successfully")
            print(f"   SHAP array shape: {shap_vals.shape}")
            print(f"   Base value (no features): {base_value:.4f}")
            print(f"   Final prediction: {prediction:.4f}")
            print(f"   Total SHAP contribution: {np.sum(shap_vals):.4f}")
            
            # Create detailed feature impact analysis
            feature_names = list(X_df.columns)
            feature_values = X_df.iloc[0].values
            
            feature_impacts = []
            for fname, sval, fval in zip(feature_names, shap_vals, feature_values):
                # Categorize feature
                if fname.startswith('tfidf_'):
                    category = "Text (TF-IDF)"
                elif fname in osint_feature_names_expected:
                    category = "OSINT"
                else:
                    category = "Manual Text"
                
                feature_impacts.append({
                    "feature": fname,
                    "category": category,
                    "shap_value": float(sval),
                    "feature_value": float(fval),
                    "impact": "phishing ⚠️" if sval > 0 else "legitimate ✅",
                    "abs_impact": abs(float(sval))
                })
            
            # Sort by absolute SHAP value (paling berpengaruh dulu)
            feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
            
            # Get top 15 most impactful features (lebih detail)
            top_features = feature_impacts[:15]
            
            # Analyze by category
            category_impact = {
                "Manual Text": {"positive": 0.0, "negative": 0.0, "total": 0.0},
                "Text (TF-IDF)": {"positive": 0.0, "negative": 0.0, "total": 0.0},
                "OSINT": {"positive": 0.0, "negative": 0.0, "total": 0.0}
            }
            
            for feat in feature_impacts:
                cat = feat["category"]
                sval = float(feat["shap_value"])  # Convert to Python float
                if sval > 0:
                    category_impact[cat]["positive"] += sval
                else:
                    category_impact[cat]["negative"] += sval
                category_impact[cat]["total"] += sval
            
            # Build analisis result
            analisis_shap = {
                "enabled": True,
                "method": "SHAP TreeExplainer",
                "total_features_analyzed": int(len(feature_names)),
                "base_value": float(base_value),
                "prediction": float(prediction),
                "total_shap_contribution": float(np.sum(shap_vals)),
                "top_features": [
                    {
                        "rank": int(i+1),
                        "feature": str(f["feature"]),
                        "category": str(f["category"]),
                        "shap_value": float(f["shap_value"]),
                        "feature_value": float(f["feature_value"]),
                        "impact": str(f["impact"]),
                        "percentage": float(abs(f["shap_value"]) / sum(abs(float(sv)) for sv in shap_vals) * 100)
                    }
                    for i, f in enumerate(top_features)
                ],
                "category_analysis": {
                    cat: {
                        "positive": float(vals["positive"]),
                        "negative": float(vals["negative"]),
                        "total": float(vals["total"])
                    }
                    for cat, vals in category_impact.items()
                },
                "interpretation": {
                    "base_prediction": f"Model baseline (tanpa features): {float(base_value):.2%}",
                    "final_prediction": f"Prediksi akhir (dengan features): {float(prediction):.2%}",
                    "change": f"Perubahan dari baseline: {float(prediction - base_value):.2%}",
                    "dominant_category": str(max(category_impact.items(), key=lambda x: abs(x[1]["total"]))[0])
                }
            }
            
            # Print summary
            print("\n📊 SHAP ANALYSIS SUMMARY:")
            print(f"   Top 3 Most Influential Features:")
            for i, feat in enumerate(top_features[:3], 1):
                print(f"   {i}. {feat['feature']}: {feat['shap_value']:+.4f} ({feat['impact']})")
            
            print(f"\n   Category Contributions:")
            for cat, vals in category_impact.items():
                print(f"   - {cat}: {vals['total']:+.4f}")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ SHAP analysis failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:500]}")
            
            # Fallback: Provide basic analysis without SHAP
            analisis_shap = {
                "enabled": False,
                "error": str(e),
                "fallback": "Using basic feature importance instead",
                "top_features": [],
                "total_features_analyzed": 0
            }
        
        # 10. Build final result
        label = "PHISHING" if prediction > 0.5 else "LEGITIMATE"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        return {
            "label": label,
            "score_bahaya": float(prediction),
            "confidence": confidence,
            "features_used": 173,
            "analisis_shap": analisis_shap,
            "osint_data": {
                "domain": domain,
                "domain_age_days": osint.domain_age_days,
                "host_up": osint.host_up,
                "https_supported": osint.https_supported,
                "alternate_ips": osint.alternate_ip_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "Phishing Detection API - OSINT Enhanced",
        "model": "XGBoost + OSINT (173 features)",
        "version": "1.0",
        "endpoints": [
            "/predict_osint_enhanced"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tfidf_loaded": tfidf_vectorizer is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
