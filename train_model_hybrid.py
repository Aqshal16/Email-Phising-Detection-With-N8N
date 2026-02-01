"""
Hybrid Training Script - Support Text-Only & OSINT-Enhanced Training
Comparison study sesuai paper An et al. (2025)

Modes:
1. TEXT_ONLY: Training dengan text features saja (baseline)
2. OSINT_ENHANCED: Training dengan text + 17 OSINT features (paper approach)

Paper results (target):
- XGBoost Text-Only: 95.39% accuracy
- XGBoost + OSINT: 96.71% accuracy (+1.32% improvement)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import re
from urllib.parse import urlparse
import os
import json
import pickle

# ========================================
# KONFIGURASI
# ========================================
MODE = "OSINT_ENHANCED"  # Options: "TEXT_ONLY" or "OSINT_ENHANCED"

# TF-IDF Configuration
USE_TFIDF = True  # Enable TF-IDF for TEXT_ONLY mode
TFIDF_MAX_FEATURES = 150  # Top 150 most important words
TFIDF_MIN_DF = 2  # Minimum document frequency
TFIDF_MAX_DF = 0.8  # Maximum document frequency (ignore too common words)

# Dataset paths
TEXT_DATASET = "dataset/phishing_with_osint_real.csv"  # Use SAME dataset for fair comparison
OSINT_DATASET = "dataset/phishing_with_osint_real.csv"  # OSINT-enhanced dataset (REAL tools: WHOIS+Nmap+theHarvester)

# Model output paths
TEXT_MODEL_PATH = "models/xgb_text_only.json"
OSINT_MODEL_PATH = "models/xgb_osint_enhanced.json"

# Results output
RESULTS_FILE = "results/training_results_hybrid.json"

# ========================================
# TEXT FEATURE EXTRACTION (BASELINE)
# ========================================

def extract_urls_from_text(text):
    """Extract URLs from email text."""
    if pd.isna(text):
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

def count_suspicious_tld(domains):
    """Count suspicious TLDs in domains."""
    suspicious_tlds = ['.xyz', '.top', '.club', '.work', '.download', '.loan', 
                       '.win', '.bid', '.click', '.stream', '.gq', '.cf', '.ml']
    count = 0
    for domain in domains:
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                count += 1
                break
    return count

def extract_text_features(text):
    """
    Extract text-based features (no OSINT, no simulation).
    Pure text analysis only - REAL FEATURES.
    
    Features:
    1. n_urls - Number of URLs in email
    2. n_domains - Number of unique domains
    3. n_ips - Number of IP addresses
    4. has_https - Has HTTPS URLs
    5. has_http - Has HTTP URLs (insecure)
    6. n_suspicious_keywords - Phishing keywords count
    7. email_length - Character count
    8. n_attachments - Attachment mentions
    9. count_suspicious_tld - Suspicious TLD count
    """
    
    features = {
        'n_urls': 0,
        'n_domains': 0,
        'n_ips': 0,
        'has_https': 0,
        'has_http': 0,
        'n_suspicious_keywords': 0,
        'email_length': 0,
        'n_attachments': 0,
        'count_suspicious_tld': 0
    }
    
    if pd.isna(text):
        return features
    
    text_str = str(text)
    
    # URL analysis
    urls = extract_urls_from_text(text_str)
    features['n_urls'] = len(urls)
    
    # Domain analysis
    domains = extract_domains_from_urls(urls)
    features['n_domains'] = len(domains)
    features['count_suspicious_tld'] = count_suspicious_tld(domains)
    
    # Protocol analysis
    features['has_https'] = 1 if any('https://' in url for url in urls) else 0
    features['has_http'] = 1 if any('http://' in url for url in urls) else 0
    
    # IP address detection
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ips = re.findall(ip_pattern, text_str)
    features['n_ips'] = len(ips)
    
    # Suspicious keywords
    suspicious_keywords = [
        'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
        'password', 'update', 'secure', 'bank', 'login', 'credit',
        'ssn', 'social security', 'expire', 'limited time', 'act now',
        'congratulations', 'winner', 'prize', 'claim', 'refund'
    ]
    text_lower = text_str.lower()
    features['n_suspicious_keywords'] = sum(1 for kw in suspicious_keywords if kw in text_lower)
    
    # Email length
    features['email_length'] = len(text_str)
    
    # Attachment mentions
    attachment_keywords = ['attachment', 'attached', 'file', 'download', 'pdf', 'doc', 'zip']
    features['n_attachments'] = sum(1 for kw in attachment_keywords if kw in text_lower)
    
    return features

# ========================================
# DATA LOADING & PREPROCESSING
# ========================================

def load_dataset(mode):
    """Load dataset based on mode with TF-IDF support."""
    
    if mode == "TEXT_ONLY":
        print(f"📂 Loading TEXT-ONLY dataset: {TEXT_DATASET}")
        df = pd.read_csv(TEXT_DATASET)
        
        # Extract manual text features (9 features)
        print("🔍 Extracting manual text features...")
        text_features = []
        for idx, row in df.iterrows():
            features = extract_text_features(row['text_combined'])
            text_features.append(features)
            
            if (idx + 1) % 5000 == 0:
                print(f"   ✓ Processed {idx + 1}/{len(df)} rows...")
        
        df_manual = pd.DataFrame(text_features)
        
        # Add TF-IDF features if enabled
        if USE_TFIDF:
            print(f"\n📊 Extracting TF-IDF features (top {TFIDF_MAX_FEATURES} words)...")
            
            # Initialize TF-IDF vectorizer
            tfidf = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Fit and transform text
            tfidf_matrix = tfidf.fit_transform(df['text_combined'].fillna(''))
            
            # Convert to DataFrame
            tfidf_feature_names = [f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
            df_tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=tfidf_feature_names
            )
            
            # Save TF-IDF vectorizer for later use
            os.makedirs('models', exist_ok=True)
            with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf, f)
            print(f"   ✓ TF-IDF vectorizer saved: models/tfidf_vectorizer.pkl")
            
            # Combine manual + TF-IDF features
            df_features = pd.concat([df_manual, df_tfidf], axis=1)
            print(f"   ✓ Combined features: {len(df_manual.columns)} manual + {len(df_tfidf.columns)} TF-IDF = {len(df_features.columns)} total")
        else:
            df_features = df_manual
            print(f"   ⚠️ TF-IDF disabled, using only {len(df_manual.columns)} manual features")
        
        # Feature columns
        feature_cols = list(df_features.columns)
        
        return df, df_features, feature_cols
    
    elif mode == "OSINT_ENHANCED":
        print(f"📂 Loading OSINT-ENHANCED dataset: {OSINT_DATASET}")
        df = pd.read_csv(OSINT_DATASET)
        
        # Extract manual text features (9 features)
        print("🔍 Extracting manual text features...")
        text_features = []
        for idx, row in df.iterrows():
            features = extract_text_features(row['text_combined'])
            text_features.append(features)
            
            if (idx + 1) % 1000 == 0:
                print(f"   ✓ Processed {idx + 1}/{len(df)} rows...")
        
        df_manual = pd.DataFrame(text_features)
        
        # Add TF-IDF features if enabled
        if USE_TFIDF:
            print(f"\n📊 Extracting TF-IDF features (top {TFIDF_MAX_FEATURES} words)...")
            
            # ALWAYS create NEW TF-IDF vectorizer for OSINT mode (no reuse)
            print(f"   ✓ Creating NEW TF-IDF vectorizer for OSINT_ENHANCED mode...")
            tfidf_path = 'models/tfidf_vectorizer.pkl'
            
            # Initialize TF-IDF vectorizer
            tfidf = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='unicode'
            )
            # Fit and transform text
            tfidf_matrix = tfidf.fit_transform(df['text_combined'].fillna(''))
            
            # Save TF-IDF vectorizer
            os.makedirs('models', exist_ok=True)
            with open(tfidf_path, 'wb') as f:
                pickle.dump(tfidf, f)
            print(f"   ✓ TF-IDF vectorizer saved: {tfidf_path}")
            
            # Convert to DataFrame
            tfidf_feature_names = [f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
            df_tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=tfidf_feature_names
            )
            
            print(f"   ✓ TF-IDF extraction complete: {len(df_tfidf.columns)} features")
        else:
            df_tfidf = pd.DataFrame()
        
        # OSINT features (14 numeric features)
        osint_feature_cols = [
            # WHOIS (1 numeric - registrar will be encoded separately)
            'domain_age_days',
            # Nmap (7 numeric features)
            'host_up', 'common_web_ports_open', 'open_ports_count', 
            'filtered_ports_count', 'https_supported', 'latency', 'scan_duration',
            # theHarvester (4 features)
            'alternate_ip_count', 'asn_found', 'host_found', 'ip_found',
            # Pattern/Helper (1 feature)
            'interesting_url'
        ]
        
        # Handle categorical feature: registrar (one-hot encode or label encode)
        if 'registrar' in df.columns:
            # Simple approach: encode as binary (has registrar or not)
            df['has_registrar'] = (df['registrar'] != 'Unknown').astype(int)
            osint_feature_cols.append('has_registrar')
        
        # Fill missing OSINT values with 0
        for col in osint_feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Combine: Manual (9) + TF-IDF (150) + OSINT (14) features
        if USE_TFIDF and not df_tfidf.empty:
            df_features = pd.concat([df_manual, df_tfidf], axis=1)
            # Add OSINT features
            for col in osint_feature_cols:
                if col in df.columns:
                    df_features[col] = df[col]
            
            print(f"\n✅ Feature combination complete:")
            print(f"   • Manual text features: {len(df_manual.columns)}")
            print(f"   • TF-IDF features: {len(df_tfidf.columns)}")
            print(f"   • OSINT features: {len(osint_feature_cols)}")
            print(f"   • Total features: {len(df_features.columns)}")
        else:
            # Fallback: Manual + OSINT only
            df_features = df_manual.copy()
            for col in osint_feature_cols:
                if col in df.columns:
                    df_features[col] = df[col]
            print(f"   ⚠️ TF-IDF disabled, using {len(df_features.columns)} features")
        
        feature_cols = list(df_features.columns)
        
        return df, df_features, feature_cols
    
    else:
        raise ValueError(f"Invalid mode: {mode}")

# ========================================
# MODEL TRAINING
# ========================================

def train_model(X, y, feature_cols, mode, hostname_column=None):
    """Train XGBoost model with GROUP-BASED SPLIT to prevent data leakage."""
    
    print(f"\n{'='*70}")
    print(f"  🚀 TRAINING MODEL - {mode} MODE")
    print(f"{'='*70}")
    
    # GROUP-BASED SPLIT by hostname (if hostname available)
    if hostname_column is not None and 'hostname' in X.columns:
        print(f"\n⚠️  Using GROUP-BASED SPLIT by hostname to prevent data leakage")
        
        # Get unique hostnames
        unique_hostnames = X['hostname'].dropna().unique()
        print(f"   Total unique hostnames: {len(unique_hostnames)}")
        
        # Create hostname to label mapping (majority vote)
        hostname_labels = X[['hostname']].copy()
        hostname_labels['label'] = y
        hostname_label_map = hostname_labels.groupby('hostname')['label'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        
        # Split hostnames by label for stratification
        safe_hostnames = [h for h in unique_hostnames if hostname_label_map.get(h, 0) == 0]
        phish_hostnames = [h for h in unique_hostnames if hostname_label_map.get(h, 1) == 1]
        
        print(f"   Safe hostnames: {len(safe_hostnames)}")
        print(f"   Phishing hostnames: {len(phish_hostnames)}")
        
        # Split hostnames 70:30
        from sklearn.model_selection import train_test_split as split_hosts
        train_safe, test_safe = split_hosts(safe_hostnames, test_size=0.30, random_state=42)
        train_phish, test_phish = split_hosts(phish_hostnames, test_size=0.30, random_state=42)
        
        train_hostnames = set(train_safe + train_phish)
        test_hostnames = set(test_safe + test_phish)
        
        # Create train/test sets based on hostname
        train_mask = X['hostname'].isin(train_hostnames)
        test_mask = X['hostname'].isin(test_hostnames)
        
        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Remove hostname from features (it's just for grouping, not for training)
        if 'hostname' in X_train.columns:
            X_train = X_train.drop(columns=['hostname'])
            X_test = X_test.drop(columns=['hostname'])
        
        # Verify no hostname overlap
        train_hosts_check = set(X[train_mask]['hostname'].unique()) if 'hostname' in X.columns else set()
        test_hosts_check = set(X[test_mask]['hostname'].unique()) if 'hostname' in X.columns else set()
        overlap = train_hosts_check.intersection(test_hosts_check)
        print(f"   ✓ Hostname overlap: {len(overlap)} (should be 0)")
        
    else:
        # Regular random split
        print(f"\n📊 Using RANDOM SPLIT")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train)} samples ({len(y_train[y_train==1])} phishing, {len(y_train[y_train==0])} safe)")
    print(f"   Testing:  {len(X_test)} samples ({len(y_test[y_test==1])} phishing, {len(y_test[y_test==0])} safe)")
    print(f"   Features: {len(feature_cols)}")
    
    # XGBoost hyperparameters (sesuai paper - Table 3)
    params = {
        'n_estimators': 100,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    print(f"\n⚙️  Training XGBoost...")
    print(f"   Hyperparameters: {params}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Results
    results = {
        'mode': mode,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_names': feature_cols
    }
    
    print(f"\n📈 RESULTS - {mode}:")
    print(f"{'─'*70}")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Save model
    model_path = TEXT_MODEL_PATH if mode == "TEXT_ONLY" else OSINT_MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return model, results

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    print("="*70)
    print("  🎯 HYBRID TRAINING - TEXT-ONLY vs OSINT-ENHANCED")
    print("="*70)
    print(f"\n📄 Paper: An et al. (2025) - Multilingual Email Phishing Detection")
    print(f"🎯 Target: XGBoost Text-Only (95.39%) → OSINT (96.71%)")
    print(f"\n🔧 Current mode: {MODE}\n")
    
    # Load dataset
    df, df_features, feature_cols = load_dataset(MODE)
    
    # Prepare data (keep hostname for grouping - BOTH modes use group-based split for fairness)
    if 'hostname' in df.columns:
        # Include hostname for grouping (will be removed in train_model)
        X = df_features.copy()
        if 'hostname' not in X.columns:
            X['hostname'] = df['hostname'].values
        hostname_column = 'hostname'
        print(f"\n✅ Using GROUP-BASED SPLIT by hostname (prevent data leakage)")
    else:
        X = df_features.copy()
        hostname_column = None
        print(f"\n⚠️ No hostname column, using random split")
    
    y = df['label'].values
    
    # Train model with hostname for group-based split
    model, results = train_model(X, y, feature_cols, MODE, hostname_column=hostname_column)
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Load existing results if any
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results[MODE] = results
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Results saved: {RESULTS_FILE}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE - {MODE}")
    print(f"{'='*70}")
    
    print(f"\n💡 Next steps:")
    if MODE == "TEXT_ONLY":
        print(f"   1. Change MODE = 'OSINT_ENHANCED' in script")
        print(f"   2. Run: python train_model_hybrid.py")
        print(f"   3. Compare: python compare_models_paper.py")
    else:
        print(f"   1. Run comparison: python compare_models_paper.py")
        print(f"   2. Generate visualizations")
        print(f"   3. Analyze feature importance with SHAP")

if __name__ == "__main__":
    main()
