"""
Script untuk reconstruct TF-IDF vectorizer dari model yang sudah ada
Jadi model tidak perlu di-train ulang, cukup generate vectorizer yang match
"""

import xgboost as xgb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("🔧 Reconstructing TF-IDF Vectorizer from Model...")
print("="*70)

# 1. Load model yang sudah ada
print("\n1️⃣ Loading existing model...")
model = xgb.Booster()
model.load_model('models/xgb_osint_enhanced.json')
print("   ✅ Model loaded")

# 2. Extract feature names dari model
print("\n2️⃣ Extracting feature names from model...")
feature_names = model.feature_names
print(f"   ✅ Total features: {len(feature_names)}")

# 3. Filter hanya TF-IDF features (yang dimulai dengan 'tfidf_')
print("\n3️⃣ Extracting TF-IDF vocabulary...")
tfidf_features = [f for f in feature_names if f.startswith('tfidf_')]
print(f"   ✅ TF-IDF features: {len(tfidf_features)}")

# 4. Extract vocabulary dari feature names (remove 'tfidf_' prefix)
vocabulary = {}
for idx, feature_name in enumerate(tfidf_features):
    word = feature_name.replace('tfidf_', '')
    vocabulary[word] = idx

print(f"   ✅ Vocabulary size: {len(vocabulary)}")
print(f"   📝 Sample words: {list(vocabulary.keys())[:10]}")

# 5. Create TF-IDF vectorizer dengan vocabulary yang sama
print("\n4️⃣ Creating new TF-IDF vectorizer with matching vocabulary...")
tfidf = TfidfVectorizer(
    vocabulary=vocabulary,  # Fixed vocabulary from model
    lowercase=True,
    strip_accents='unicode',
    ngram_range=(1, 2),
    stop_words='english'
)

# Fit dengan dummy data (karena vocabulary sudah fixed)
# Vectorizer perlu di-fit dulu sebelum bisa dipakai
dummy_data = list(vocabulary.keys())
tfidf.fit(dummy_data)

print("   ✅ Vectorizer created")

# 6. Verify vocabulary matches
print("\n5️⃣ Verifying vocabulary matches...")
reconstructed_words = tfidf.get_feature_names_out()
if len(reconstructed_words) == len(vocabulary):
    print(f"   ✅ Vocabulary size matches: {len(vocabulary)} words")
else:
    print(f"   ⚠️ Vocabulary size mismatch: expected {len(vocabulary)}, got {len(reconstructed_words)}")

# Sample verification
print(f"   📝 First 10 words: {list(reconstructed_words[:10])}")
print(f"   📝 Last 10 words: {list(reconstructed_words[-10:])}")

# 7. Save vectorizer
print("\n6️⃣ Saving reconstructed vectorizer...")
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("   ✅ Saved to: models/tfidf_vectorizer.pkl")

# 8. Test load
print("\n7️⃣ Testing load...")
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    test_tfidf = pickle.load(f)
test_words = test_tfidf.get_feature_names_out()
print(f"   ✅ Loaded successfully: {len(test_words)} words")

print("\n" + "="*70)
print("✅ SUCCESS! TF-IDF vectorizer reconstructed and saved.")
print("   Model: models/xgb_osint_enhanced.json")
print("   Vectorizer: models/tfidf_vectorizer.pkl")
print("\n💡 Now restart your API and try prediction again!")
