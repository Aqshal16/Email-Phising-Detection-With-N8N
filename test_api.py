"""
Test script untuk API endpoint
"""

import requests
import json

# Test data
test_email = """
Dear Customer,

Your account has been suspended due to suspicious activity. 
Click here to verify your account: http://verify-account.xyz/login

Please update your password immediately at https://secure-banking.top/account/password

Best regards,
Customer Service
"""

test_osint_features = {
    "domain_age_days": 30,
    "has_registrar": 0,
    "host_up": 0,
    "common_web_ports_open": 0,
    "open_ports_count": 0,
    "filtered_ports_count": 0,
    "https_supported": 0,
    "latency": 999.0,
    "scan_duration": 5.0,
    "alternate_ip_count": 0,
    "asn_found": 0,
    "host_found": 0,
    "ip_found": 0,
    "interesting_url": 1
}

# Make request
try:
    print("🔍 Testing API endpoint: /predict_osint_enhanced")
    print(f"{'='*70}")
    
    response = requests.post(
        "http://localhost:8000/predict_osint_enhanced",
        json={
            "email_text": test_email,
            "osint_features": test_osint_features
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        print(f"\n✅ API is working!")
    else:
        print(f"Error response:")
        print(response.text)
        print(f"\n❌ API returned error")
    
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API. Is it running?")
    print("   Run: python api_osint_enhanced.py")
except Exception as e:
    print(f"❌ Error: {e}")
