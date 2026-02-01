"""
Script Extract OSINT Features dengan REAL TOOLS
Tools: WHOIS + Nmap + theHarvester (Paper-Authentic)

Paper: An et al. (2025) - Multilingual Email Phishing Detection
17 OSINT Features extracted using actual OSINT tools
"""

import pandas as pd
import re
import socket
import subprocess
import json
import time
import whois
from datetime import datetime
from urllib.parse import urlparse
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET

# ========================================
# KONFIGURASI
# ========================================
DATASET_FILE = "dataset/phishing_with_osint_paper.csv"  # 1500 samples with URLs
OUTPUT_FILE = "dataset/phishing_with_osint_real.csv"
CHECKPOINT_FILE = "dataset/osint_checkpoint.csv"

MAX_SAMPLES = 450  # Extract 450 MORE samples (target total: ~1000 samples)
CHECKPOINT_INTERVAL = 10  # Save every 10 samples

# Tool configuration
USE_WHOIS = True
USE_NMAP = True  
USE_THEHARVESTER = True  # theHarvester installed as Python module

# Timeouts
WHOIS_TIMEOUT = 10
NMAP_TIMEOUT = 60
HARVESTER_TIMEOUT = 45

# ========================================
# 1. WHOIS Lookup (Domain Age, Registrar)
# ========================================
def get_whois_info(domain):
    """
    REAL WHOIS lookup using python-whois
    Returns: domain_age_days, registrar, creation_date
    """
    if not USE_WHOIS:
        return {'domain_age_days': -1, 'registrar': 'Unknown', 'whois_found': 0}
    
    try:
        w = whois.whois(domain)
        
        # Extract creation date
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        
        # Calculate domain age
        if creation_date:
            age_days = (datetime.now() - creation_date).days
        else:
            age_days = -1
        
        registrar = w.registrar if w.registrar else "Unknown"
        
        return {
            'domain_age_days': age_days,
            'registrar': registrar,
            'creation_date': str(creation_date) if creation_date else None,
            'whois_found': 1
        }
    except Exception as e:
        return {
            'domain_age_days': -1,
            'registrar': 'Unknown', 
            'creation_date': None,
            'whois_found': 0
        }

# ========================================
# 2. Nmap Scan (Real Port Scanning)
# ========================================
def run_nmap_scan(domain):
    """
    REAL Nmap scan for port detection
    Command: nmap -Pn -T4 --top-ports 100 -oX - <domain>
    """
    if not USE_NMAP:
        return {
            'host_up': 0, 'open_ports_count': 0, 'filtered_ports_count': 0,
            'common_web_ports_open': 0, 'https_supported': 0,
            'open_ports': [], 'services': [], 'latency': -1, 'scan_duration': 0
        }
    
    start_time = time.time()
    
    try:
        cmd = [
            'nmap',
            '-Pn',  # Skip ping
            '-T4',  # Aggressive timing
            '--top-ports', '100',  # Scan 100 most common ports
            '--max-retries', '2',
            '-oX', '-',  # XML output to stdout
            domain
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=NMAP_TIMEOUT
        )
        
        # Parse XML output
        root = ET.fromstring(result.stdout)
        host = root.find('host')
        
        if host is None:
            return {
                'host_up': 0, 'open_ports_count': 0, 'filtered_ports_count': 0,
                'common_web_ports_open': 0, 'https_supported': 0,
                'open_ports': [], 'services': [], 'latency': -1,
                'scan_duration': time.time() - start_time
            }
        
        # Host status
        status = host.find('status')
        host_up = 1 if status is not None and status.get('state') == 'up' else 0
        
        # Latency
        times = host.find('times')
        latency = float(times.get('srtt')) / 1000000 if times is not None else -1
        
        # Parse ports
        ports = host.find('ports')
        open_ports = []
        services = []
        filtered_count = 0
        
        if ports is not None:
            for port in ports.findall('port'):
                port_id = port.get('portid')
                state = port.find('state')
                service = port.find('service')
                
                if state is not None:
                    state_val = state.get('state')
                    if state_val == 'open':
                        open_ports.append(int(port_id))
                        if service is not None:
                            services.append(service.get('name', 'unknown'))
                    elif state_val == 'filtered':
                        filtered_count += 1
        
        # Check common web ports
        web_ports = {80, 443, 8080, 8443}
        common_web_open = len(web_ports.intersection(set(open_ports)))
        https_supported = 1 if 443 in open_ports else 0
        
        return {
            'host_up': host_up,
            'open_ports_count': len(open_ports),
            'filtered_ports_count': filtered_count,
            'common_web_ports_open': common_web_open,
            'https_supported': https_supported,
            'open_ports': open_ports,
            'services': services,
            'latency': latency,
            'scan_duration': time.time() - start_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            'host_up': 0, 'open_ports_count': 0, 'filtered_ports_count': 0,
            'common_web_ports_open': 0, 'https_supported': 0,
            'open_ports': [], 'services': [], 'latency': -1,
            'scan_duration': NMAP_TIMEOUT
        }
    except Exception as e:
        print(f"Nmap error for {domain}: {str(e)}")
        return {
            'host_up': 0, 'open_ports_count': 0, 'filtered_ports_count': 0,
            'common_web_ports_open': 0, 'https_supported': 0,
            'open_ports': [], 'services': [], 'latency': -1,
            'scan_duration': time.time() - start_time
        }

# ========================================
# 3. theHarvester (OSINT Gathering)
# ========================================
def run_theharvester(domain):
    """
    REAL theHarvester for OSINT gathering using CLI
    Runs theHarvester command and parses output
    """
    if not USE_THEHARVESTER:
        return {
            'alternate_ip_count': 0, 'ip_address': None,
            'asn_found': 0, 'host_found': 0
        }
    
    try:
        # Run theHarvester CLI command using venv executable
        venv_harvester = os.path.join(os.getcwd(), '.venv', 'Scripts', 'theHarvester.exe')
        
        # Use venv path if exists, otherwise try system command
        harvester_cmd = venv_harvester if os.path.exists(venv_harvester) else 'theHarvester'
        
        cmd = [
            harvester_cmd,
            '-d', domain,
            '-b', 'hackertarget',  # Free source, no API key required
            '-l', '50'  # Limit results
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=HARVESTER_TIMEOUT,
            cwd=os.getcwd()  # Run from current directory
        )
        
        output = result.stdout + result.stderr
        
        # Extract IPs from output
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, output)
        # Filter valid IPs
        valid_ips = [ip for ip in ips if all(0 <= int(octet) <= 255 for octet in ip.split('.'))]
        alternate_ip_count = len(set(valid_ips))
        
        # Check ASN in output
        asn_found = 1 if 'ASN' in output or re.search(r'AS\d+', output) else 0
        
        # Check hosts found
        host_found = 1 if 'Hosts found' in output or len(valid_ips) > 0 else 0
        
        # Get primary IP
        ip_address = valid_ips[0] if valid_ips else None
        
        # Clean up any temp files created by theHarvester
        try:
            for f in os.listdir('.'):
                if f.startswith('temp_') or f.endswith('.xml') or f.endswith('.json'):
                    if domain.replace('.', '_') in f:
                        try:
                            os.remove(f)
                        except:
                            pass
        except:
            pass
        
        return {
            'alternate_ip_count': alternate_ip_count,
            'ip_address': ip_address,
            'asn_found': asn_found,
            'host_found': host_found
        }
        
    except subprocess.TimeoutExpired:
        # Fallback to DNS lookup on timeout
        try:
            ip = socket.gethostbyname(domain)
            return {
                'alternate_ip_count': 1,
                'ip_address': ip,
                'asn_found': 0,
                'host_found': 1
            }
        except:
            return {
                'alternate_ip_count': 0, 'ip_address': None,
                'asn_found': 0, 'host_found': 0
            }
    except Exception as e:
        # Fallback to DNS lookup on any error
        try:
            ip = socket.gethostbyname(domain)
            return {
                'alternate_ip_count': 1,
                'ip_address': ip,
                'asn_found': 0,
                'host_found': 1
            }
        except:
            return {
                'alternate_ip_count': 0, 'ip_address': None,
                'asn_found': 0, 'host_found': 0
            }

# ========================================
# HELPER FUNCTIONS
# ========================================
def extract_urls_from_text(text):
    """Extract URLs from email text (handles preprocessed text without spaces)"""
    if pd.isna(text):
        return []
    
    text_str = str(text)
    urls = []
    
    # Try standard URL pattern first
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls.extend(re.findall(url_pattern, text_str))
    
    # If no URLs, try preprocessed patterns like 'httpwalllaidcom' or 'wwwgooglecom'
    if not urls:
        # Pattern: http + word + com/org/net (no spaces/dots)
        preprocessed_url = r'(?:http|https|www)([a-z0-9]+)(?:com|org|net|edu|gov|co|us|uk|de|fr|info)'
        matches = re.findall(preprocessed_url, text_str, re.IGNORECASE)
        for match in matches:
            urls.append(f'http://{match}.com')
    
    # Try to find domain patterns with dots: word.com, www.word.com
    if not urls:
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, text_str)
        urls = [f'http://{d}' for d in domains]
    
    return urls

def extract_domains_from_urls(urls):
    """Extract unique domains from URLs"""
    domains = []
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain and not domain.replace('.', '').replace('-', '').isdigit():
                domains.append(domain)
        except:
            pass
    return list(set(domains))

def extract_ips_from_text(text):
    """Extract IP addresses from text"""
    if pd.isna(text):
        return []
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ips = re.findall(ip_pattern, str(text))
    return [ip for ip in ips if all(0 <= int(octet) <= 255 for octet in ip.split('.'))]

def check_rdns(ip):
    """Get reverse DNS for IP"""
    try:
        return socket.gethostbyaddr(ip)[0]
    except:
        return None

def check_suspicious_url(url):
    """Check if URL contains suspicious keywords"""
    suspicious = [
        'login', 'verify', 'account', 'update', 'secure', 'banking',
        'paypal', 'ebay', 'confirm', 'suspended', 'password', 'urgent'
    ]
    url_lower = url.lower()
    return 1 if any(keyword in url_lower for keyword in suspicious) else 0

# ========================================
# MAIN EXTRACTION FUNCTION
# ========================================
def extract_osint_features_real(text):
    """
    Extract 17 OSINT features using REAL TOOLS
    
    Features:
    1. hostname - Primary domain
    2. host_up - Host reachability (Nmap)
    3. alternate_ip_count - Alternative IPs (theHarvester)
    4. ip_address - Primary IP (theHarvester)
    5. common_web_ports_open - Web ports (Nmap)
    6. open_ports_count - Total open ports (Nmap)
    7. filtered_ports_count - Filtered ports (Nmap)
    8. open_ports - List of open ports (Nmap)
    9. rdns_record - Reverse DNS
    10. https_supported - HTTPS support (Nmap 443)
    11. services - Running services (Nmap)
    12. host_found - Host detected (theHarvester)
    13. interesting_url - Suspicious URL keywords
    14. asn_found - ASN detected (theHarvester)
    15. ip_found - Email contains IP
    16. latency - Response delay (Nmap)
    17. scan_duration - Total scan time (Nmap)
    + domain_age_days - Domain age (WHOIS)
    + registrar - Domain registrar (WHOIS)
    """
    
    # Initialize features
    features = {
        'hostname': None,
        'host_up': 0,
        'alternate_ip_count': 0,
        'ip_address': None,
        'common_web_ports_open': 0,
        'open_ports_count': 0,
        'filtered_ports_count': 0,
        'open_ports': [],
        'rdns_record': None,
        'https_supported': 0,
        'services': [],
        'host_found': 0,
        'interesting_url': 0,
        'asn_found': 0,
        'ip_found': 0,
        'latency': -1,
        'scan_duration': 0,
        'domain_age_days': -1,
        'registrar': 'Unknown'
    }
    
    # Extract URLs and domains
    urls = extract_urls_from_text(text)
    if not urls:
        return features
    
    domains = extract_domains_from_urls(urls)
    if not domains:
        return features
    
    primary_domain = domains[0]
    features['hostname'] = primary_domain
    
    # Check for IPs in text
    ips_in_text = extract_ips_from_text(text)
    features['ip_found'] = 1 if ips_in_text else 0
    
    # Check suspicious URL
    features['interesting_url'] = max([check_suspicious_url(url) for url in urls])
    
    # 1. WHOIS lookup
    whois_info = get_whois_info(primary_domain)
    features['domain_age_days'] = whois_info['domain_age_days']
    features['registrar'] = whois_info['registrar']
    
    # 2. Nmap scan
    nmap_results = run_nmap_scan(primary_domain)
    features['host_up'] = nmap_results['host_up']
    features['open_ports_count'] = nmap_results['open_ports_count']
    features['filtered_ports_count'] = nmap_results['filtered_ports_count']
    features['common_web_ports_open'] = nmap_results['common_web_ports_open']
    features['https_supported'] = nmap_results['https_supported']
    features['open_ports'] = nmap_results['open_ports']
    features['services'] = nmap_results['services']
    features['latency'] = nmap_results['latency']
    features['scan_duration'] = nmap_results['scan_duration']
    
    # 3. theHarvester
    harvester_results = run_theharvester(primary_domain)
    features['alternate_ip_count'] = harvester_results['alternate_ip_count']
    features['ip_address'] = harvester_results['ip_address']
    features['asn_found'] = harvester_results['asn_found']
    features['host_found'] = harvester_results['host_found']
    
    # 4. Reverse DNS (if we have IP)
    if features['ip_address']:
        features['rdns_record'] = check_rdns(features['ip_address'])
    
    return features

# ========================================
# MAIN EXECUTION
# ========================================
def main():
    print("="*60)
    print("OSINT Feature Extraction - REAL TOOLS")
    print("Tools: WHOIS + Nmap + theHarvester")
    print("="*60)
    
    # Check if tools are available
    print("\n[1/4] Checking tool availability...")
    
    # Check Nmap
    try:
        subprocess.run(['nmap', '--version'], capture_output=True, timeout=5)
        print("✓ Nmap detected")
    except:
        print("✗ Nmap NOT FOUND - Please install from https://nmap.org/download.html")
        if USE_NMAP:
            return
    
    # Check theHarvester
    try:
        # Try venv path first
        venv_harvester = os.path.join(os.getcwd(), '.venv', 'Scripts', 'theHarvester.exe')
        if os.path.exists(venv_harvester):
            print("✓ theHarvester detected (venv)")
        else:
            # Try system command
            subprocess.run(['theHarvester', '--help'], capture_output=True, timeout=5)
            print("✓ theHarvester detected")
    except:
        print("✗ theHarvester NOT FOUND")
        print("  Note: Will use DNS fallback for IP/host detection")
        print("  To install: pip install theHarvester")
        global USE_THEHARVESTER
        USE_THEHARVESTER = False
    
    # Check WHOIS
    try:
        import whois
        print("✓ python-whois detected")
    except:
        print("✗ python-whois NOT FOUND - Install: pip install python-whois")
        if USE_WHOIS:
            return
    
    # Load dataset
    print(f"\n[2/4] Loading dataset: {DATASET_FILE}")
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: Dataset not found: {DATASET_FILE}")
        return
    
    df = pd.read_csv(DATASET_FILE)
    print(f"Total emails: {len(df)}")
    print(f"Processing {MAX_SAMPLES} samples...")
    
    # Check for checkpoint OR existing output file
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n[*] Checkpoint found! Loading from {CHECKPOINT_FILE}")
        df_checkpoint = pd.read_csv(CHECKPOINT_FILE)
        start_idx = len(df_checkpoint)
        print(f"Resuming from sample {start_idx + 1}")
    elif os.path.exists(OUTPUT_FILE):
        print(f"\n[*] Existing output found! Loading from {OUTPUT_FILE}")
        df_existing = pd.read_csv(OUTPUT_FILE)
        start_idx = len(df_existing)
        print(f"Already processed: {start_idx} samples")
        print(f"Continuing from sample {start_idx + 1}")
    
    # Select samples to process
    df_sample = df.iloc[start_idx:start_idx + MAX_SAMPLES].copy()
    
    # Extract OSINT features
    print(f"\n[3/4] Extracting OSINT features (START: {start_idx}, END: {start_idx + len(df_sample)})")
    print(f"Estimated time: ~{(len(df_sample) * 1.5) / 60:.1f} minutes")
    print("="*60)
    
    results = []
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing"):
        # Get text column (handle both 'text' and 'text_combined')
        text = row.get('text', row.get('text_combined', ''))
        
        # Extract OSINT features
        features = extract_osint_features_real(text)
        
        # Combine with original row
        result = {
            'label': row['label'],
            'text_combined': text,
            **features
        }
        results.append(result)
        
        # Checkpoint save
        if len(results) % CHECKPOINT_INTERVAL == 0:
            df_checkpoint = pd.DataFrame(results)
            if os.path.exists(CHECKPOINT_FILE):
                df_existing = pd.read_csv(CHECKPOINT_FILE)
                df_checkpoint = pd.concat([df_existing, df_checkpoint], ignore_index=True)
            df_checkpoint.to_csv(CHECKPOINT_FILE, index=False)
            print(f"\n[CHECKPOINT] Saved {len(df_checkpoint)} samples")
    
    # Save final results
    print(f"\n[4/4] Saving results to {OUTPUT_FILE}")
    df_final = pd.DataFrame(results)
    
    # Merge with existing checkpoint or output file
    if os.path.exists(CHECKPOINT_FILE):
        df_existing = pd.read_csv(CHECKPOINT_FILE)
        df_final = pd.concat([df_existing, df_final], ignore_index=True)
    elif os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_final = pd.concat([df_existing, df_final], ignore_index=True)
        print(f"[*] Merged with existing {len(df_existing)} samples")
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # Remove checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("[*] Checkpoint file removed")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print(f"Total samples: {len(df_final)}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)
    
    # Show sample
    print("\nSample features:")
    print(df_final[['hostname', 'host_up', 'open_ports_count', 'domain_age_days']].head())

if __name__ == "__main__":
    main()
