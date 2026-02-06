import json
from eth_account import Account
from eth_account.messages import encode_defunct
import time
from datetime import datetime, timezone, timedelta
import random
import os
import sys
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Use curl_cffi for TLS fingerprint bypass (fixes 403 errors)
try:
    from curl_cffi import requests as curl_requests
    USE_CURL_CFFI = True
except ImportError:
    import requests as curl_requests
    USE_CURL_CFFI = False
    print("⚠️ curl_cffi not installed, using requests (may get 403 errors)")

# Thread lock for file operations and console output
file_lock = threading.Lock()
print_lock = threading.Lock()

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Browser impersonation options for curl_cffi (only use supported versions)
BROWSER_IMPERSONATIONS = [
    "chrome110",
    "chrome107",
    "chrome104",
    "chrome101",
    "chrome100",
    "chrome99",
]

def create_session_with_tls_bypass(proxy=None):
    """Create a session with TLS fingerprint bypass using curl_cffi"""
    if USE_CURL_CFFI:
        # Random browser impersonation
        impersonate = random.choice(BROWSER_IMPERSONATIONS)
        session = curl_requests.Session(impersonate=impersonate)
        if proxy:
            session.proxies = proxy
        return session
    else:
        # Fallback to regular requests
        import requests
        session = requests.Session()
        if proxy:
            session.proxies = proxy
        return session

# Get script directory for relative file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(SCRIPT_DIR, 'config.json')
    try:
        if not os.path.exists(config_path):
            default_config = {
                "HUGGINGFACE_API_KEY": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print("⚠️  config.json created. Please edit with your HuggingFace API key.")
            return default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Config error: {e}")
        return {"HUGGINGFACE_API_KEY": None}


def load_proxies(filename="proxy.txt"):
    """Load proxies from file"""
    proxy_path = os.path.join(SCRIPT_DIR, filename)
    proxies = []
    try:
        if not os.path.exists(proxy_path):
            return []
        
        with open(proxy_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    proxies.append(line)
        
        return proxies
    except Exception as e:
        print(f"[-] Error loading proxies: {e}")
        return []


def parse_proxy(proxy_string):
    """Parse proxy string into requests format
    Supports formats:
    - user:pass@host:port
    - host:port:user:pass
    - host:port
    - http://user:pass@host:port
    """
    try:
        proxy = proxy_string.strip()
        if proxy.startswith('http://'):
            proxy = proxy[7:]
        elif proxy.startswith('https://'):
            proxy = proxy[8:]
        
        if '@' in proxy:
            auth, hostport = proxy.rsplit('@', 1)
            return {
                'http': f'http://{auth}@{hostport}',
                'https': f'http://{auth}@{hostport}'
            }
        
        parts = proxy.split(':')
        if len(parts) == 4:
            host, port, user, password = parts
            return {
                'http': f'http://{user}:{password}@{host}:{port}',
                'https': f'http://{user}:{password}@{host}:{port}'
            }
        elif len(parts) == 2:
            return {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}'
            }
        else:
            return None
    except Exception:
        return None


def mask_proxy(proxy):
    """Mask proxy credentials for display"""
    if not proxy:
        return "None"
    if '@' in proxy:
        host_part = proxy.split('@')[-1]
        return f"***@{host_part}"
    return proxy[:25] + '...' if len(proxy) > 25 else proxy


def get_next_daily_reset():
    """Get next daily reset time (00:00 UTC)"""
    now = datetime.now(timezone.utc)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return tomorrow


def format_countdown(seconds):
    """Format seconds into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def wait_with_countdown(target_time, message="Next daily reset"):
    """Wait until target time with live countdown"""
    while True:
        now = datetime.now(timezone.utc)
        remaining = (target_time - now).total_seconds()
        
        if remaining <= 0:
            print("\n")
            return
        
        countdown = format_countdown(remaining)
        sys.stdout.write(f"\r⏳ {message}: {countdown} remaining   ")
        sys.stdout.flush()
        time.sleep(1)


def load_wallets(source="wallet"):
    """Load wallets from wallet.json or accounts.json
    
    Args:
        source: "wallet" for wallet.json, "accounts" for accounts.json, "both" for combined
    """
    wallets = []
    
    if source in ["wallet", "both"]:
        wallet_path = os.path.join(SCRIPT_DIR, 'wallet.json')
        try:
            if os.path.exists(wallet_path):
                with open(wallet_path, 'r') as f:
                    wallets_data = json.load(f)
                
                for idx, wallet_obj in enumerate(wallets_data, 1):
                    for key, info in wallet_obj.items():
                        wallets.append({
                            'name': key,
                            'address': info['address'],
                            'private_key': info['private_key'],
                            'source': 'wallet.json'
                        })
        except Exception as e:
            print(f"❌ Error loading wallet.json: {e}")
    
    if source in ["accounts", "both"]:
        accounts_path = os.path.join(SCRIPT_DIR, 'accounts.json')
        try:
            if os.path.exists(accounts_path):
                with open(accounts_path, 'r') as f:
                    accounts_data = json.load(f)
                
                for idx, account in enumerate(accounts_data, 1):
                    if 'wallet' in account:
                        wallet_info = account['wallet']
                        # Use referral_code as name if available, otherwise use index
                        name = account.get('referral_code', f'account{idx}')
                        wallets.append({
                            'name': name,
                            'address': wallet_info['address'],
                            'private_key': wallet_info['private_key'],
                            'source': 'accounts.json'
                        })
        except Exception as e:
            print(f"❌ Error loading accounts.json: {e}")
    
    # Remove duplicates based on address (keep first occurrence)
    seen_addresses = set()
    unique_wallets = []
    for wallet in wallets:
        addr_lower = wallet['address'].lower()
        if addr_lower not in seen_addresses:
            seen_addresses.add(addr_lower)
            unique_wallets.append(wallet)
    
    return unique_wallets




# ============================================================
# HELPER CLASSES
# ============================================================

class TwitterUsernameGenerator:
    """Generate realistic Twitter usernames"""
    FIRST_NAMES = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
        "Thomas", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Andrew",
        "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward"]
    
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"]
    
    SUFFIXES = ["Dev", "Tech", "Code", "Crypto", "Web3", "NFT", "DeFi", "AI", "Pro", "Elite"]

    @staticmethod
    def generate():
        first = random.choice(TwitterUsernameGenerator.FIRST_NAMES)
        last = random.choice(TwitterUsernameGenerator.LAST_NAMES)
        suffix = random.choice(TwitterUsernameGenerator.SUFFIXES)
        num = random.randint(1, 9999)
        
        pattern = random.choice(["fl", "fs", "ls", "fln", "fn"])
        if pattern == "fl": return f"{first}{last}"
        elif pattern == "fs": return f"{first}{suffix}"
        elif pattern == "ls": return f"{last}{suffix}"
        elif pattern == "fln": return f"{first}{last}{num}"
        else: return f"{first}{num}"


class GlobalSubmittedTracker:
    """Track all submitted items per wallet address globally (thread-safe)"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, tracker_file="submitted_tracker.json"):
        if self._initialized:
            return
        self.tracker_file = os.path.join(SCRIPT_DIR, tracker_file)
        self.tracker = self._load_tracker()
        self._initialized = True
    
    def _load_tracker(self):
        """Load tracker from file"""
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_tracker(self):
        """Save tracker to file (thread-safe)"""
        try:
            with file_lock:
                with open(self.tracker_file, 'w') as f:
                    json.dump(self.tracker, f, indent=2)
        except Exception:
            pass
    
    def get_submitted_items(self, wallet_address):
        """Get all submitted items for a specific wallet"""
        addr_lower = wallet_address.lower()
        if addr_lower not in self.tracker:
            self.tracker[addr_lower] = {"models": [], "datasets": []}
        return self.tracker[addr_lower]
    
    def add_submitted_item(self, wallet_address, item_name, item_type="model"):
        """Add a submitted item for a wallet"""
        addr_lower = wallet_address.lower()
        if addr_lower not in self.tracker:
            self.tracker[addr_lower] = {"models": [], "datasets": []}
        
        key = f"{item_type}s"
        if item_name not in self.tracker[addr_lower][key]:
            self.tracker[addr_lower][key].append(item_name)
            self._save_tracker()
    
    def is_item_submitted(self, wallet_address, item_name, item_type="model"):
        """Check if an item was already submitted by this wallet"""
        addr_lower = wallet_address.lower()
        if addr_lower not in self.tracker:
            return False
        key = f"{item_type}s"
        return item_name in self.tracker[addr_lower].get(key, [])
    
    def sync_from_api(self, wallet_address, models_list, datasets_list):
        """Sync submitted items from API response"""
        addr_lower = wallet_address.lower()
        if addr_lower not in self.tracker:
            self.tracker[addr_lower] = {"models": [], "datasets": []}
        
        # Merge with existing
        for model in models_list:
            if model not in self.tracker[addr_lower]["models"]:
                self.tracker[addr_lower]["models"].append(model)
        for dataset in datasets_list:
            if dataset not in self.tracker[addr_lower]["datasets"]:
                self.tracker[addr_lower]["datasets"].append(dataset)
        
        self._save_tracker()


class SuccessAnalytics:
    """Analytics for tracking and learning from successful submissions"""
    def __init__(self, history_file="success_history.json"):
        self.history_file = os.path.join(SCRIPT_DIR, history_file)
        self.history = self.load_history()
    
    def load_history(self):
        """Load success history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            "models": {
                "successful": [],
                "failed": [],
                "blacklist": []  # Items that failed too many times
            },
            "datasets": {
                "successful": [],
                "failed": [],
                "blacklist": []
            }
        }
    
    def save_history(self):
        """Save history to file (thread-safe)"""
        try:
            with file_lock:
                with open(self.history_file, 'w') as f:
                    json.dump(self.history, f, indent=2)
        except Exception as e:
            pass
    
    def add_success(self, item_data, points=0):
        """Record successful submission"""
        item_type = item_data['fileType']
        key = f"{item_type}s"  # 'models' or 'datasets'
        
        # Add to successful list (avoid duplicates)
        name = item_data['fileName']
        if name not in [s['name'] for s in self.history[key]['successful']]:
            self.history[key]['successful'].append({
                "name": name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "points": points
            })
        
        # Remove from failed/blacklist if present
        if name in self.history[key]['failed']:
            self.history[key]['failed'].remove(name)
        if name in self.history[key]['blacklist']:
            self.history[key]['blacklist'].remove(name)
        
        self.save_history()
    
    def add_failure(self, item_data, error_msg=""):
        """Record failed submission"""
        item_type = item_data['fileType']
        key = f"{item_type}s"
        name = item_data['fileName']
        
        # Add to failed list if not already there
        if name not in self.history[key]['failed']:
            self.history[key]['failed'].append(name)
        
        # Blacklist if fails too many times (appears 3+ times in failed list)
        fail_count = self.history[key]['failed'].count(name)
        if fail_count >= 3 and name not in self.history[key]['blacklist']:
            self.history[key]['blacklist'].append(name)
        
        self.save_history()
    
    def get_successful_names(self, item_type="model"):
        """Get list of successful item names"""
        key = f"{item_type}s"
        return [s['name'] for s in self.history[key]['successful']]
    
    def get_successful_authors(self, item_type="model"):
        """Get list of authors who had successful submissions"""
        successful = self.get_successful_names(item_type)
        authors = set()
        for name in successful:
            if '/' in name:
                authors.add(name.split('/')[0])
        return list(authors)
    
    def is_blacklisted(self, name, item_type="model"):
        """Check if item is blacklisted"""
        key = f"{item_type}s"
        return name in self.history[key]['blacklist']
    
    def get_similar_to_successful(self, item_type="model", pool=None):
        """Find items similar to past successes"""
        if not pool:
            return None
        
        successful = self.get_successful_names(item_type)
        if not successful:
            return None
        
        # Find items from same authors as successful ones
        authors = self.get_successful_authors(item_type)
        similar_items = []
        
        for item in pool:
            if '/' in item:
                author = item.split('/')[0]
                if author in authors:
                    similar_items.append(item)
        
        return similar_items if similar_items else None


class BrowserFingerprint:
    """Generate browser fingerprints"""
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    ]
    
    CHROME_VERSIONS = ["144"]

    @staticmethod
    def generate():
        # Use Edge v144 as seen in successful API capture
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0"
        
        platform = "Windows"
        # Match exact format from API capture
        browser_header = '"Not(A:Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"'
        
        return {
            "user_agent": ua,
            "sec_ch_ua": browser_header,
            "sec_ch_ua_mobile": "?0",
            "sec_ch_ua_platform": f'"{platform}"',
            "accept_language": "en-US,en;q=0.9"
        }


# ============================================================
# ACCOUNT CREATION BOT
# ============================================================

class HumanoidNetwork:
    """Humanoid Network Account Creator"""
    def __init__(self, fingerprint=None, proxy=None, referral_code=None):
        self.base_url = "https://app.humanoidnetwork.org/api"
        self.wallet = None
        self.account = None
        self.token = None
        self.user_data = None
        self.fingerprint = fingerprint or BrowserFingerprint.generate()
        self.referral_code = referral_code  # Store for referer header
        
        # Parse and setup proxy
        parsed_proxy = None
        if proxy:
            parsed_proxy = parse_proxy(proxy)
        
        # Use curl_cffi session with TLS bypass
        self.session = create_session_with_tls_bypass(parsed_proxy)

    def get_headers(self, include_auth=False):
        """Generate headers with fingerprint - matches exact API capture format"""
        headers = {
            "accept": "*/*",
            "accept-language": self.fingerprint["accept_language"],
            "cache-control": "no-cache",
            "content-type": "application/json",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": self.fingerprint["sec_ch_ua"],
            "sec-ch-ua-mobile": self.fingerprint["sec_ch_ua_mobile"],
            "sec-ch-ua-platform": self.fingerprint["sec_ch_ua_platform"],
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.fingerprint["user_agent"]
        }
        # Add referer header with referral code if available
        if self.referral_code:
            headers["referer"] = f"https://app.humanoidnetwork.org/ref/{self.referral_code}"
        if include_auth and self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def generate_wallet(self):
        """Generate new Ethereum wallet"""
        print("[*] Generating new wallet...")
        Account.enable_unaudited_hdwallet_features()
        self.account = Account.create()
        self.wallet = {
            'address': self.account.address,
            'private_key': self.account.key.hex()
        }
        print(f"[+] Wallet generated: {self.wallet['address']}")
        return self.wallet

    def sign_message(self, message):
        """Sign message with private key"""
        try:
            signable_message = encode_defunct(text=message)
            signed_message = self.account.sign_message(signable_message)
            signature = signed_message.signature.hex()
            if not signature.startswith('0x'):
                signature = '0x' + signature
            return signature
        except Exception:
            return None

    def get_nonce(self):
        """Get nonce for authentication"""
        print("[*] Getting nonce...")
        try:
            response = self.session.post(f"{self.base_url}/auth/nonce", 
                json={"walletAddress": self.wallet['address']}, headers=self.get_headers())
            if response.status_code == 200:
                data = response.json()
                print(f"[+] Nonce received: {data['nonce']}")
                return data
            else:
                print(f"[-] Failed to get nonce: {response.status_code}")
                return None
        except Exception as e:
            print(f"[-] Exception getting nonce: {e}")
            return None

    def authenticate(self, nonce_data, referral_code, max_retries=3):
        """Authenticate with signature and referral code"""
        print("[*] Authenticating...")
        signature = self.sign_message(nonce_data['message'])
        if not signature:
            print("[-] Failed to sign message")
            return None
        
        payload = {
            "walletAddress": self.wallet['address'],
            "signature": signature,
            "message": nonce_data['message'],
            "referralCode": referral_code
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(f"{self.base_url}/auth/authenticate", 
                    json=payload, headers=self.get_headers())
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        self.token = data['token']
                        self.user_data = data['user']
                        print(f"[+] Authentication successful!")
                        print(f"[+] User ID: {self.user_data['id']}")
                        print(f"[+] Referral Code: {self.user_data.get('referralCode', 'N/A')}")
                        return data
                    else:
                        print(f"[-] Authentication failed: {data}")
                        return None
                elif response.status_code == 500:
                    print(f"[-] Server error (500). Attempt {attempt}/{max_retries}")
                    if attempt < max_retries:
                        wait_time = 5 * (2 ** (attempt - 1))
                        print(f"[*] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        new_nonce = self.get_nonce()
                        if new_nonce:
                            nonce_data = new_nonce
                            signature = self.sign_message(nonce_data['message'])
                            if signature:
                                payload['signature'] = signature
                                payload['message'] = nonce_data['message']
                    else:
                        print(f"[-] Max retries reached.")
                        return None
                else:
                    print(f"[-] Authentication error: {response.status_code}")
                    print(f"[-] Response: {response.text}")
                    return None
            except Exception as e:
                print(f"[-] Exception: {e}")
                if attempt >= max_retries:
                    return None
                time.sleep(5)
        return None

    def get_user_info(self):
        """Get user information"""
        try:
            response = self.session.get(f"{self.base_url}/user", headers=self.get_headers(include_auth=True))
            if response.status_code == 200:
                data = response.json()
                self.user_data = data['user']
                return data
            return None
        except Exception:
            return None

    def get_tasks(self):
        """Get available tasks"""
        try:
            response = self.session.get(f"{self.base_url}/tasks", headers=self.get_headers(include_auth=True))
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def complete_task(self, task):
        """Complete a task"""
        try:
            payload = {"taskId": task['id'], "data": task.get('requirements', {})}
            response = self.session.post(f"{self.base_url}/tasks", json=payload, 
                headers=self.get_headers(include_auth=True))
            return response.status_code == 200
        except Exception:
            return False

    def clear_tasks(self):
        """Complete all available tasks"""
        print("[*] Completing tasks...")
        tasks = self.get_tasks()
        if not tasks:
            print("[-] No tasks available")
            return
        
        completed = 0
        for task in tasks:
            if self.complete_task(task):
                print(f"[+] Completed: {task['title']} (+{task['points']} pts)")
                completed += 1
                time.sleep(2)
        
        print(f"[+] Completed {completed}/{len(tasks)} tasks")

    def save_account_data(self, wallet_number=1):
        """Save wallet to wallet.json only (thread-safe)"""
        with print_lock:
            print("[*] Saving wallet data...")
        
        wallet_filepath = os.path.join(SCRIPT_DIR, "wallet.json")
        
        with file_lock:
            try:
                with open(wallet_filepath, 'r') as f:
                    wallets = json.load(f)
            except FileNotFoundError:
                wallets = []
            
            # Get next wallet number based on existing wallets
            next_num = len(wallets) + 1
            
            wallet_entry = {
                f"wallet{next_num}": {
                    "address": self.wallet['address'],
                    "private_key": self.wallet['private_key']
                }
            }
            
            wallets.append(wallet_entry)
            
            with open(wallet_filepath, 'w') as f:
                json.dump(wallets, f, indent=2)
        
        with print_lock:
            print(f"[+] Wallet saved to wallet.json (wallet{next_num})")
        
        # Get user info for return data
        user_info = self.get_user_info()
        return {
            "wallet": self.wallet,
            "user_data": user_info,
            "referral_code": self.user_data.get('referralCode', 'N/A')
        }


# ============================================================
# TRAINING BOT
# ============================================================

class HumanoidTraining:
    """Humanoid Network Training Bot"""
    
    # Fallback models/datasets if API fetch fails - 100+ popular models
    FALLBACK_MODELS = [
        # Meta Llama models
        "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct",
        # Qwen models
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-72B-Instruct", "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen-VL-Chat", "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct",
        # Google models
        "google/gemma-2b", "google/gemma-7b", "google/gemma-2-9b-it", "google/gemma-2-27b-it",
        "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl",
        "google/t5-v1_1-base", "google/t5-v1_1-large", "google/pegasus-xsum",
        # Mistral models
        "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Codestral-22B-v0.1",
        # Microsoft models
        "microsoft/phi-2", "microsoft/phi-1_5", "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct", "microsoft/DialoGPT-medium",
        "microsoft/codebert-base", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large",
        # OpenAI/GPT models
        "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small",
        "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14",
        # Stability AI
        "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stablelm-tuned-alpha-7b",
        "stabilityai/stable-code-3b", "stabilityai/stablelm-2-12b-chat",
        # Hugging Face models
        "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-alpha",
        "HuggingFaceM4/idefics-80b-instruct", "HuggingFaceM4/idefics2-8b",
        # NVIDIA
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "nvidia/Mistral-NeMo-12B-Instruct",
        # DeepSeek
        "deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-ai/deepseek-llm-7b-chat",
        "deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/deepseek-coder-33b-instruct",
        # Alibaba
        "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-base-en-v1.5",
        # Yi models
        "01-ai/Yi-6B-Chat", "01-ai/Yi-34B-Chat", "01-ai/Yi-1.5-9B-Chat",
        # Sentence transformers
        "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # BERT variants
        "bert-base-uncased", "bert-large-uncased", "bert-base-multilingual-cased",
        "roberta-base", "roberta-large", "distilbert-base-uncased",
        # Other popular
        "facebook/opt-1.3b", "facebook/opt-6.7b", "facebook/bart-large-cnn",
        "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B", "EleutherAI/pythia-6.9b",
        "bigscience/bloom-560m", "bigscience/bloom-1b7", "bigscience/bloom-7b1",
        "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct",
        "teknium/OpenHermes-2.5-Mistral-7B", "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "TheBloke/Llama-2-7B-Chat-GGUF", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat", "baichuan-inc/Baichuan2-7B-Chat",
        "internlm/internlm2-chat-7b", "upstage/SOLAR-10.7B-Instruct-v1.0",
        "lmsys/vicuna-7b-v1.5", "lmsys/vicuna-13b-v1.5",
        "WizardLM/WizardLM-7B-V1.0", "WizardLM/WizardCoder-Python-7B-V1.0",
        "codellama/CodeLlama-7b-Instruct-hf", "codellama/CodeLlama-34b-Instruct-hf",
        "Salesforce/codegen-2B-mono", "Salesforce/blip2-opt-2.7b",
        "mosaicml/mpt-7b-instruct", "databricks/dolly-v2-7b",
    ]
    
    FALLBACK_DATASETS = [
        # Text datasets
        "HuggingFaceFW/fineweb-edu", "allenai/c4", "wikimedia/wikipedia",
        "squad", "glue", "super_glue", "hellaswag", "winogrande",
        "allenai/ai2_arc", "cais/mmlu", "openai/gsm8k", "lighteval/mmlu",
        "tatsu-lab/alpaca", "databricks/dolly-15k", "OpenAssistant/oasst1",
        "HuggingFaceH4/ultrachat_200k", "stingning/ultrachat", 
        # Code datasets
        "bigcode/starcoderdata", "codeparrot/github-code", "bigcode/the-stack",
        "sahil2801/CodeAlpaca-20k", "TokenBender/code_instructions_120k",
        # Multilingual
        "facebook/multilingual_librispeech", "google/fleurs",
        "csebuetnlp/xlsum", "Helsinki-NLP/opus-100",
        # Image datasets
        "laion/laion2B-en", "ChristophSchuhmann/improved_aesthetics_6.5plus",
        "poloclub/diffusiondb", "lambdalabs/pokemon-blip-captions",
        # Conversation
        "HuggingFaceH4/no_robots", "teknium/OpenHermes-2.5",
        "Open-Orca/OpenOrca", "WizardLM/WizardLM_evol_instruct_V2_196k",
        # Math & Science
        "lighteval/MATH", "camel-ai/math", "meta-math/MetaMathQA",
        "Rowan/hellaswag", "ceval/ceval-exam",
        # QA datasets
        "natural_questions", "trivia_qa", "ms_marco",
        "hotpot_qa", "PY007/TinyStoriesV2", "Anthropic/hh-rlhf",
        # Summarization
        "cnn_dailymail", "xsum", "multi_news", "billsum",
        # Translation
        "wmt14", "wmt16", "iwslt2017", "opus_books",
        # Classification
        "imdb", "yelp_review_full", "ag_news", "amazon_reviews_multi",
        "hate_speech18", "sentiment140", "tweet_eval",
        # Other popular
        "lmsys/chatbot_arena_conversations", "nvidia/ChatQA-Training-Data",
        "TIGER-Lab/MathInstruct", "m-a-p/CodeFeedback-Filtered-Instruction",
        "HuggingFaceTB/cosmopedia", "Skywork/Skywork-Reward-Preference-80K",
        "argilla/OpenHermesPreferences", "Intel/orca_dpo_pairs",
    ]

    def __init__(self, wallet_data, fingerprint=None, hf_api_key=None, proxy=None):
        self.base_url = "https://app.humanoidnetwork.org/api"
        self.wallet = wallet_data
        self.fingerprint = fingerprint or BrowserFingerprint.generate()
        self.hf_api_key = hf_api_key
        self.account = Account.from_key(self.wallet['private_key'])
        self.token = None
        self.user_data = None
        self.submitted_items = set()  # Local tracking for current session
        self.total_points = 0
        
        # Parse and setup proxy
        parsed_proxy = None
        if proxy:
            parsed_proxy = parse_proxy(proxy)
        
        # Dynamic model and dataset pools (auto-fetched)
        self.MODELS_POOL = []
        self.DATASETS_POOL = []
        
        # Use curl_cffi session with TLS bypass
        self.session = create_session_with_tls_bypass(parsed_proxy)
        
        # Initialize success analytics and global tracker
        self.analytics = SuccessAnalytics()
        self.global_tracker = GlobalSubmittedTracker()
        
        # Auto-fetch trending models and datasets from HuggingFace
        self._auto_fetch_items()

    def _auto_fetch_items(self):
        """Auto-fetch trending models and datasets from HuggingFace API"""
        # Start with fallback models as base (always available)
        self.MODELS_POOL = list(self.FALLBACK_MODELS)
        self.DATASETS_POOL = list(self.FALLBACK_DATASETS)
        
        # Try to fetch more from HuggingFace API
        fetched_models = self._fetch_from_huggingface("models", limit=300)
        if fetched_models:
            # Add fetched models to pool (avoid duplicates)
            for model in fetched_models:
                if model not in self.MODELS_POOL:
                    self.MODELS_POOL.append(model)
            print(f"   ├─ 🔄 Pool: {len(self.MODELS_POOL)} models ({len(fetched_models)} from API + {len(self.FALLBACK_MODELS)} fallback)")
        else:
            print(f"   ├─ ⚠️  Using {len(self.MODELS_POOL)} fallback models (API fetch failed)")
        
        fetched_datasets = self._fetch_from_huggingface("datasets", limit=300)
        if fetched_datasets:
            # Add fetched datasets to pool (avoid duplicates)
            for dataset in fetched_datasets:
                if dataset not in self.DATASETS_POOL:
                    self.DATASETS_POOL.append(dataset)
            print(f"   ├─ 🔄 Pool: {len(self.DATASETS_POOL)} datasets ({len(fetched_datasets)} from API + {len(self.FALLBACK_DATASETS)} fallback)")
        else:
            print(f"   ├─ ⚠️  Using {len(self.DATASETS_POOL)} fallback datasets (API fetch failed)")
    
    def _fetch_from_huggingface(self, item_type="models", limit=300):
        """Fetch items from HuggingFace with multiple strategies"""
        all_items = set()  # Use set to avoid duplicates
        
        # Try multiple sorting strategies to get diverse items
        sort_options = ["trending", "downloads", "likes", "created"]
        
        for sort_by in sort_options:
            try:
                # Fetch in batches
                batch_size = 100
                for offset in range(0, min(limit, 200), batch_size):
                    url = f"https://huggingface.co/api/{item_type}?sort={sort_by}&direction=-1&limit={batch_size}&skip={offset}"
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
                    if self.hf_api_key:
                        headers["Authorization"] = f"Bearer {self.hf_api_key}"
                    
                    try:
                        import requests
                        resp = requests.get(url, headers=headers, timeout=10)
                        
                        if resp.status_code == 200:
                            items = resp.json()
                            for item in items:
                                if 'id' in item and '/' in item['id']:
                                    all_items.add(item['id'])
                        
                        # Small delay between requests
                        time.sleep(0.3)
                        
                    except Exception:
                        continue
                    
                    # If we have enough items, stop
                    if len(all_items) >= limit:
                        break
                        
            except Exception:
                continue
            
            # If we have enough items, stop trying other sort options
            if len(all_items) >= limit:
                break
        
        return list(all_items)[:limit] if all_items else []

    def get_headers(self, include_auth=False):
        """Generate headers"""
        headers = {
            "accept": "*/*",
            "accept-language": self.fingerprint["accept_language"],
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://app.humanoidnetwork.org",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://app.humanoidnetwork.org/training",
            "sec-ch-ua": self.fingerprint["sec_ch_ua"],
            "sec-ch-ua-mobile": self.fingerprint["sec_ch_ua_mobile"],
            "sec-ch-ua-platform": self.fingerprint["sec_ch_ua_platform"],
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.fingerprint["user_agent"]
        }
        if include_auth and self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def sign_message(self, message):
        """Sign message with private key"""
        try:
            signable = encode_defunct(text=message)
            signed = self.account.sign_message(signable)
            sig = signed.signature.hex()
            return sig if sig.startswith('0x') else '0x' + sig
        except Exception:
            return None

    def login(self, max_retries=3):
        """Login using wallet with retry logic for rate limiting"""
        addr_short = f"{self.wallet['address'][:8]}...{self.wallet['address'][-6:]}"
        print(f"   ├─ 🔐 Logging in: {addr_short}")
        
        for attempt in range(1, max_retries + 1):
            try:
                # Add small random delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 2.0))
                
                resp = self.session.post(f"{self.base_url}/auth/nonce", 
                    json={"walletAddress": self.wallet['address']}, headers=self.get_headers())
                
                if resp.status_code in [403, 429]:
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) + random.uniform(1, 3)
                        print(f"   ├─ ⏳ Rate limited ({resp.status_code}), retry {attempt}/{max_retries} in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    print(f"   ├─ ❌ Failed to get nonce: {resp.status_code} (max retries reached)")
                    return False
                    
                if resp.status_code != 200: 
                    print(f"   ├─ ❌ Failed to get nonce: {resp.status_code}")
                    return False
                nonce_data = resp.json()
                
                signature = self.sign_message(nonce_data['message'])
                if not signature: 
                    print(f"   ├─ ❌ Failed to sign message")
                    return False
                
                # Small delay before auth
                time.sleep(random.uniform(0.3, 1.0))
                
                resp = self.session.post(f"{self.base_url}/auth/authenticate", json={
                    "walletAddress": self.wallet['address'],
                    "signature": signature,
                    "message": nonce_data['message']
                }, headers=self.get_headers())
                
                if resp.status_code in [403, 429]:
                    if attempt < max_retries:
                        wait_time = (2 ** attempt) + random.uniform(1, 3)
                        print(f"   ├─ ⏳ Rate limited ({resp.status_code}), retry {attempt}/{max_retries} in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    print(f"   ├─ ❌ Auth failed: {resp.status_code} (max retries reached)")
                    return False
                    
                if resp.status_code != 200: 
                    print(f"   ├─ ❌ Auth failed: {resp.status_code}")
                    return False
                auth_data = resp.json()
                
                if auth_data.get('success'):
                    self.token = auth_data['token']
                    self.user_data = auth_data['user']
                    print(f"   ├─ ✅ Login successful (ID: {self.user_data['id'][:12]}...)")
                    self._auto_twitter()
                    self._load_submitted()
                    return True
                
                print(f"   ├─ ❌ Auth failed: {auth_data}")
                return False
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    print(f"   ├─ ⚠️ Error: {str(e)[:50]}, retry {attempt}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                print(f"   ├─ ❌ Login error: {e}")
                return False
        
        return False

    def get_tasks(self):
        """Get available tasks"""
        try:
            resp = self.session.get(f"{self.base_url}/tasks", headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_user_task_completions(self):
        """Get user's completed tasks"""
        try:
            resp = self.session.get(f"{self.base_url}/user", headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                data = resp.json()
                return [tc['taskId'] for tc in data.get('user', {}).get('taskCompletions', [])]
            return []
        except Exception:
            return []

    def complete_task(self, task):
        """Complete a task"""
        try:
            requirements = task.get('requirements', {})
            payload = {
                "taskId": task['id'],
                "data": requirements
            }
            resp = self.session.post(f"{self.base_url}/tasks", json=payload, 
                headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def clear_all_tasks(self, show_output=True):
        """Complete all available tasks"""
        tasks = self.get_tasks()
        if not tasks:
            if show_output:
                print(f"   ├─ 📋 No tasks available")
            return 0
        
        completed_ids = self.get_user_task_completions()
        pending_tasks = [t for t in tasks if t['id'] not in completed_ids]
        
        if not pending_tasks:
            if show_output:
                print(f"   ├─ 📋 All {len(tasks)} tasks already completed")
            return 0
        
        if show_output:
            print(f"   ├─ 📋 Clearing {len(pending_tasks)} tasks...")
        
        completed = 0
        total_points = 0
        for task in pending_tasks:
            result = self.complete_task(task)
            if result:
                completed += 1
                total_points += task['points']
                if show_output:
                    print(f"   │  └─ ✅ {task['title'][:30]} (+{task['points']} pts)")
            else:
                if show_output:
                    print(f"   │  └─ ❌ {task['title'][:30]}")
            time.sleep(1)
        
        if show_output:
            print(f"   ├─ 📋 Tasks: {completed}/{len(pending_tasks)} completed (+{total_points} pts)")
        return total_points

    def _auto_twitter(self):
        """Automatically check and set Twitter username - DISABLED"""
        # DISABLED: Auto Twitter username setting has been turned off
        # if self.user_data and self.user_data.get('twitterId'):
        #     print(f"   ├─ 🐦 Twitter: @{self.user_data['twitterId']} (exists)")
        #     return
        # 
        # username = TwitterUsernameGenerator.generate()
        # try:
        #     resp = self.session.post(f"{self.base_url}/user/update-x-username",
        #         json={"twitterUsername": username}, headers=self.get_headers(include_auth=True))
        #     if resp.status_code == 200 and resp.json().get('success'):
        #         print(f"   ├─ 🐦 Twitter: @{username} (new)")
        #         if self.user_data: 
        #             self.user_data['twitterId'] = username
        #     else:
        #         print(f"   ├─ 🐦 Twitter: ❌ Failed to set")
        # except Exception as e:
        #     print(f"   ├─ 🐦 Twitter: ❌ Error: {e}")
        pass

    def _load_submitted(self):
        """Load already submitted items from API and sync to global tracker"""
        try:
            resp = self.session.get(f"{self.base_url}/training", headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                trainings = resp.json()
                
                # Collect models and datasets separately for global tracker
                models_list = []
                datasets_list = []
                
                for t in trainings:
                    self.submitted_items.add(t['fileName'])
                    self.submitted_items.add(t['fileUrl'])
                    
                    # Track by type
                    if t['fileType'] == 'model':
                        models_list.append(t['fileName'])
                    else:
                        datasets_list.append(t['fileName'])
                
                # Sync to global tracker
                self.global_tracker.sync_from_api(
                    self.wallet['address'],
                    models_list,
                    datasets_list
                )
                
                if trainings:
                    print(f"   ├─ 📋 Previous: {len(models_list)} models, {len(datasets_list)} datasets")
        except Exception as e:
            print(f"   ├─ 📋 Previous: ❌ {e}")

    def get_progress(self):
        """Get training progress"""
        try:
            resp = self.session.get(f"{self.base_url}/training/progress", headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                return resp.json()
            return None
        except:
            return None

    def get_random_item(self, item_type="model"):
        """Smart selection: prioritize NEW items not yet submitted by this wallet"""
        pool = self.MODELS_POOL if item_type == "model" else self.DATASETS_POOL
        wallet_address = self.wallet['address']
        
        # Get items already submitted by THIS wallet from global tracker
        wallet_submitted = self.global_tracker.get_submitted_items(wallet_address)
        already_submitted = set(wallet_submitted.get(f"{item_type}s", []))
        
        # Filter out already submitted items (by this wallet) and blacklisted items
        available = []
        for item_name in pool:
            # Skip if already submitted by this wallet
            if item_name in already_submitted:
                continue
            
            # Skip if in current session submitted items
            if item_name in self.submitted_items:
                continue
            
            # Skip blacklisted items
            if self.analytics.is_blacklisted(item_name, item_type):
                continue
            
            available.append(item_name)
        
        # If all items from pool are submitted, we have a problem - fetch more or warn
        if not available:
            print(f"   │     └─ ⚠️  All {len(pool)} {item_type}s exhausted for this wallet!")
            # Try to fetch more items
            new_pool = self._fetch_from_huggingface(
                "models" if item_type == "model" else "datasets",
                limit=100
            )
            if new_pool:
                # Add new items to pool
                if item_type == "model":
                    self.MODELS_POOL.extend([x for x in new_pool if x not in self.MODELS_POOL])
                    pool = self.MODELS_POOL
                else:
                    self.DATASETS_POOL.extend([x for x in new_pool if x not in self.DATASETS_POOL])
                    pool = self.DATASETS_POOL
                
                # Re-filter available
                available = [x for x in new_pool if x not in already_submitted 
                            and x not in self.submitted_items 
                            and not self.analytics.is_blacklisted(x, item_type)]
            
            # Still nothing? Use random from original pool as last resort
            if not available:
                available = [x for x in pool if x not in self.submitted_items]
                if not available:
                    available = pool  # Ultimate fallback
        
        # SMART SELECTION ALGORITHM - prioritize items from successful authors
        name = None
        
        # 60% - Try items from same authors as past successful submissions
        if random.random() < 0.60:
            similar = self.analytics.get_similar_to_successful(item_type, available)
            if similar:
                name = random.choice(similar)
        
        # 40% - Random exploration from available pool (for variety)
        if not name:
            name = random.choice(available)
        
        # Construct proper URL
        if item_type == "model":
            fileUrl = f"https://huggingface.co/{name}"
        else:
            fileUrl = f"https://huggingface.co/datasets/{name}"
        
        return {"fileName": name, "fileUrl": fileUrl, "fileType": item_type}

    def get_description(self, item_name, item_type="model"):
        """Get description from HuggingFace"""
        try:
            api_url = f"https://huggingface.co/api/{'models' if item_type == 'model' else 'datasets'}/{item_name}"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
            resp = requests.get(api_url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if 'cardData' in data and data['cardData']:
                    desc = data['cardData'].get('description') or data['cardData'].get('dataset_description')
                    if desc: return desc[:500]
                
                short_name = item_name.split('/')[-1]
                if item_type == "model":
                    pipeline = data.get('pipeline_tag', 'machine learning')
                    return f"A {pipeline} model called {short_name}. Provides high-quality AI outputs."
                return f"A dataset called {short_name}. Contains quality training data for AI models."
        except: pass
        
        short_name = item_name.split('/')[-1]
        return f"A {'model' if item_type == 'model' else 'dataset'} called {short_name} for AI training."

    def submit_item(self, item_data, description):
        """Submit training item"""
        # Check if already submitted (either by fileName or fileUrl)
        if item_data['fileName'] in self.submitted_items or item_data['fileUrl'] in self.submitted_items:
            return None
        
        payload = {
            "fileName": item_data['fileName'],
            "fileUrl": item_data['fileUrl'],
            "fileType": item_data['fileType'],
            "recaptchaToken": ""  # API doesn't require captcha
        }
        
        try:
            resp = self.session.post(f"{self.base_url}/training", json=payload, 
                headers=self.get_headers(include_auth=True))
            
            if resp.status_code == 200:
                data = resp.json()
                points = data.get('points', 0)
                self.total_points += points
                self.submitted_items.add(item_data['fileName'])
                self.submitted_items.add(item_data['fileUrl'])
                
                # Track success in analytics
                self.analytics.add_success(item_data, points)
                
                # Track in global tracker for this wallet
                self.global_tracker.add_submitted_item(
                    self.wallet['address'],
                    item_data['fileName'],
                    item_data['fileType']
                )
                
                return data
            else:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get('error') or error_data.get('message') or resp.text
                except:
                    error_msg = resp.text or str(resp.status_code)
                
                # Track failure in analytics
                self.analytics.add_failure(item_data, error_msg)
                
                # Only add to submitted if it's a permanent error (already submitted, invalid, etc)
                if "already" in error_msg.lower() or "invalid" in error_msg.lower():
                    self.submitted_items.add(item_data['fileName'])
                    self.submitted_items.add(item_data['fileUrl'])
                
                print(f"   │     └─ ❌ {error_msg}")
                return None
        except Exception as e:
            print(f"   │     └─ ❌ {e}")
            return None

    def do_training(self, model_count=3, dataset_count=3):
        """Perform training tasks"""
        progress = self.get_progress()
        if not progress:
            print(f"   └─ ❌ Cannot get progress")
            return False
        
        daily = progress['daily']
        models_remaining = daily['models']['remaining']
        datasets_remaining = daily['datasets']['remaining']
        
        # Show clear progress with lock status
        models_done = daily['models']['completed']
        models_limit = daily['models']['limit']
        datasets_done = daily['datasets']['completed']
        datasets_limit = daily['datasets']['limit']
        
        # Datasets are locked until 3 models completed
        dataset_status = "🔓" if models_done >= models_limit else "🔒"
        print(f"   ├─ 📊 TODAY: Models {models_done}/{models_limit} ✅ | Datasets {datasets_done}/{datasets_limit} {dataset_status}")
        
        # Check if all daily limits are already reached
        if models_remaining == 0 and datasets_remaining == 0:
            print(f"   └─ ✅ All daily limits already completed! Skipping...")
            return True
        
        actual_models = min(model_count, models_remaining)
        
        if actual_models == 0 and daily['models']['completed'] < daily['models']['limit']:
            print(f"   └─ ⚠️  Need to complete models first!")
            return True
        
        successful = 0
        
        # Step 1: Complete all models first - KEEP TRYING UNTIL 3/3 SUCCESS
        if actual_models > 0:
            print(f"   ├─ 📋 Step 1: Completing {actual_models} models...")
            
            for i in range(actual_models):
                submitted = False
                retry_count = 0
                
                # UNLIMITED RETRIES until successful
                while not submitted:
                    item = self.get_random_item("model")
                    name = item['fileName'][:35] + "..." if len(item['fileName']) > 35 else item['fileName']
                    
                    if retry_count == 0:
                        print(f"   ├─ 🤖 [{i+1}/{actual_models}] {name}")
                    else:
                        print(f"   │     └─ 🔄 Retry {retry_count}: {name}")
                    
                    desc = self.get_description(item['fileName'], "model")
                    
                    # Wait 2 seconds before submitting
                    print(f"   │     └─ ⏳ Waiting 2s before submit...")
                    time.sleep(2)
                    
                    result = self.submit_item(item, desc)
                    
                    if result:
                        print(f"   │     └─ ✅ +{result['points']} pts")
                        successful += 1
                        submitted = True
                    else:
                        retry_count += 1
                        # Exponential backoff: wait longer after more failures
                        if retry_count % 5 == 0:
                            wait = min(retry_count * 2, 10)  # Max 10 seconds
                            print(f"   │     └─ ⏳ Cooling down {wait}s...")
                            time.sleep(wait)
                        else:
                            time.sleep(2)
                
                time.sleep(random.randint(2, 4))
        else:
            print(f"   ├─ ✅ Models already completed ({daily['models']['completed']}/{daily['models']['limit']})")
        
        # Step 2: Refresh progress to check if datasets are now unlocked
        print(f"   ├─ 🔄 Refreshing progress...")
        progress = self.get_progress()
        if not progress:
            print(f"   └─ ❌ Cannot refresh progress")
            return False
        
        daily = progress['daily']
        models_completed = daily['models']['completed']
        datasets_remaining = daily['datasets']['remaining']
        actual_datasets = min(dataset_count, datasets_remaining)
        
        print(f"   ├─ 📊 Updated: Models {models_completed}/{daily['models']['limit']} | Datasets {daily['datasets']['completed']}/{daily['datasets']['limit']}")
        
        # CRITICAL: Datasets only unlock when models are 3/3
        if models_completed < daily['models']['limit']:
            print(f"   ├─ 🔒 Datasets locked - need {daily['models']['limit']}/{daily['models']['limit']} models first (currently {models_completed}/{daily['models']['limit']})")
            actual_datasets = 0
        
        # Step 3: Complete datasets (only available after models are 3/3) - KEEP TRYING UNTIL SUCCESS
        if actual_datasets > 0 and models_completed >= daily['models']['limit']:
            print(f"   ├─ 📋 Step 2: Completing {actual_datasets} datasets...")
            
            for i in range(actual_datasets):
                submitted = False
                retry_count = 0
                
                # UNLIMITED RETRIES until successful
                while not submitted:
                    item = self.get_random_item("dataset")
                    name = item['fileName'][:35] + "..." if len(item['fileName']) > 35 else item['fileName']
                    
                    if retry_count == 0:
                        print(f"   ├─ 📚 [{i+1}/{actual_datasets}] {name}")
                    else:
                        print(f"   │     └─ 🔄 Retry {retry_count}: {name}")
                    
                    desc = self.get_description(item['fileName'], "dataset")
                    
                    # Wait 2 seconds before submitting
                    print(f"   │     └─ ⏳ Waiting 2s before submit...")
                    time.sleep(2)
                    
                    result = self.submit_item(item, desc)
                    
                    if result:
                        print(f"   │     └─ ✅ +{result['points']} pts")
                        successful += 1
                        submitted = True
                    else:
                        retry_count += 1
                        # Exponential backoff: wait longer after more failures
                        if retry_count % 5 == 0:
                            wait = min(retry_count * 2, 10)  # Max 10 seconds
                            print(f"   │     └─ ⏳ Cooling down {wait}s...")
                            time.sleep(wait)
                        else:
                            time.sleep(2)
                
                if i < actual_datasets - 1:
                    time.sleep(random.randint(2, 4))
        elif daily['datasets']['completed'] >= daily['datasets']['limit']:
            print(f"   ├─ ✅ Datasets already completed ({daily['datasets']['completed']}/{daily['datasets']['limit']})")
        elif daily['models']['completed'] < daily['models']['limit']:
            print(f"   ├─ ⚠️  Datasets locked - need 3/3 models first")
        
        total = actual_models + actual_datasets
        if total == 0:
            print(f"   └─ ✅ All tasks already completed!")
        else:
            print(f"   └─ 📈 Session: {successful}/{total} | +{self.total_points} pts")
        return True


# ============================================================
# MENU FUNCTIONS
# ============================================================

def create_account_threaded(referral_code, account_number, total_accounts, proxy=None):
    """Create single account with unique fingerprint and optional proxy (thread-safe)"""
    with print_lock:
        print(f"\n{'='*60}")
        print(f"[Thread-{threading.current_thread().name}] Creating Account {account_number}/{total_accounts}")
        print(f"{'='*60}")
    
    fingerprint = BrowserFingerprint.generate()
    with print_lock:
        print(f"[{account_number}] Fingerprint: {fingerprint['user_agent'][:50]}...")
        print(f"[{account_number}] Proxy: {mask_proxy(proxy)}")
    
    bot = HumanoidNetwork(fingerprint=fingerprint, proxy=proxy, referral_code=referral_code)
    
    bot.generate_wallet()
    
    nonce_data = bot.get_nonce()
    if not nonce_data:
        with print_lock:
            print(f"[{account_number}] ❌ Failed to get nonce. Skipping...")
        return {"success": False, "account_number": account_number}
    
    auth_data = bot.authenticate(nonce_data, referral_code)
    if not auth_data:
        with print_lock:
            print(f"[{account_number}] ❌ Authentication failed. Skipping...")
        return {"success": False, "account_number": account_number}
    
    bot.get_user_info()
    bot.clear_tasks()
    account_data = bot.save_account_data()
    
    with print_lock:
        print(f"\n{'='*60}")
        print(f"[{account_number}] ✅ ACCOUNT COMPLETED!")
        print(f"[{account_number}] Wallet: {bot.wallet['address']}")
        print(f"[{account_number}] Referral Code: {bot.user_data.get('referralCode', 'N/A')}")
        print(f"{'='*60}")
    
    return {"success": True, "account_number": account_number, "data": account_data}


def menu_create_accounts():
    """Menu option 1: Create new accounts with threading"""
    print("\n" + "="*60)
    print("         📝 CREATE NEW ACCOUNTS (THREADED)")
    print("="*60)
    
    # Ask about proxy
    use_proxy = input("\nDo you want to use proxy? (y/n): ").strip().lower()
    proxies = []
    if use_proxy == 'y':
        proxies = load_proxies("proxy.txt")
        if proxies:
            print(f"[+] Loaded {len(proxies)} proxies from proxy.txt")
        else:
            print("[!] No proxies found in proxy.txt. Running without proxy.")
    else:
        print("[+] Running without proxy.")
    
    # Ask for number of accounts
    while True:
        try:
            num_accounts = int(input("\nHow many accounts to create?: "))
            if num_accounts >= 1:
                break
            print("[-] Please enter a number >= 1")
        except ValueError:
            print("[-] Please enter a valid number")
    
    # Ask for number of threads
    max_threads = min(num_accounts, 10)
    while True:
        try:
            num_threads = int(input(f"\nNumber of threads? (1-{max_threads}): "))
            if 1 <= num_threads <= max_threads:
                break
            print(f"[-] Please enter a number between 1-{max_threads}")
        except ValueError:
            print("[-] Please enter a valid number")
    
    # Ask for referral code
    referral_code = input("Enter referral code: ").strip()
    if not referral_code:
        referral_code = "WYPUMM"
    print(f"[+] Using referral code: {referral_code}")
    print(f"[+] Running with {num_threads} threads...")
    print("\n" + "="*60)
    
    # Create accounts using ThreadPoolExecutor
    successful_accounts = []
    failed_accounts = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for i in range(1, num_accounts + 1):
            proxy = proxies[(i - 1) % len(proxies)] if proxies else None
            future = executor.submit(create_account_threaded, referral_code, i, num_accounts, proxy)
            futures[future] = i
        
        for future in as_completed(futures):
            result = future.result()
            if result and result.get("success"):
                successful_accounts.append(result.get("data"))
            else:
                failed_accounts += 1
    
    # Summary
    print("\n\n" + "="*60)
    print("ALL ACCOUNTS CREATION COMPLETED!")
    print("="*60)
    print(f"Total Requested: {num_accounts}")
    print(f"Successfully Created: {len(successful_accounts)}")
    print(f"Failed: {failed_accounts}")
    print(f"Threads Used: {num_threads}")
    print(f"\nData saved to: wallet.json")
    print("="*60)


def train_wallet_threaded(wallet, idx, total_wallets, hf_api_key, proxy, max_retries=20):
    """Train a single wallet with retry logic (thread-safe worker function)"""
    wallet_name = wallet['name']
    addr_short = f"{wallet['address'][:8]}...{wallet['address'][-6:]}"
    
    with print_lock:
        print(f"\n┌─ 💼 [{idx}/{total_wallets}] Wallet: {wallet_name}")
        if proxy:
            print(f"│  🌐 Proxy: {mask_proxy(proxy)}")
        print(f"│  📍 Address: {addr_short}")
    
    # Retry loop - attempt up to max_retries times
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            with print_lock:
                print(f"│  🔄 Retry attempt {attempt}/{max_retries}")
        
        bot = HumanoidTraining(
            wallet,
            fingerprint=BrowserFingerprint.generate(),
            hf_api_key=hf_api_key,
            proxy=proxy
        )
        
        if not bot.login():
            with print_lock:
                print(f"│  ⚠️  Login failed (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2)  # Wait before retry
                continue
            else:
                with print_lock:
                    print(f"└─ [{idx}] ❌ Login failed after {max_retries} attempts\n")
                return {"success": False, "wallet_name": wallet_name, "points": 0}
        
        time.sleep(1)
        
        # Daily training - only submit AI models/datasets
        if bot.do_training(model_count=3, dataset_count=3):
            with print_lock:
                print(f"\n└─ [{idx}] ✅ Completed | +{bot.total_points} pts (attempt {attempt})")
            return {"success": True, "wallet_name": wallet_name, "points": bot.total_points}
        else:
            with print_lock:
                print(f"│  ⚠️  Training failed (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2)  # Wait before retry
                continue
            else:
                with print_lock:
                    print(f"\n└─ [{idx}] ❌ Training failed after {max_retries} attempts")
                return {"success": False, "wallet_name": wallet_name, "points": 0}
    
    # Fallback return (should not reach here)
    return {"success": False, "wallet_name": wallet_name, "points": 0}


def menu_daily_training():
    """Menu option 2: Run daily training with threading"""
    print("\n" + "="*60)
    print("         🎓 DAILY TRAINING BOT (THREADED)")
    print("="*60)
    
    config = load_config()
    
    hf_api_key = config.get("HUGGINGFACE_API_KEY")
    if not hf_api_key or hf_api_key == "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx":
        hf_api_key = None
    
    wallets = load_wallets()
    if not wallets:
        print("❌ No wallets found in wallet.json!")
        print("Please create accounts first using option 1.")
        return
    
    print(f"[+] Loaded {len(wallets)} wallets from wallet.json")
    
    # Ask about proxy
    use_proxy = input("\nDo you want to use proxy? (y/n): ").strip().lower()
    proxies = []
    if use_proxy == 'y':
        proxies = load_proxies("proxy.txt")
        if proxies:
            print(f"[+] Loaded {len(proxies)} proxies from proxy.txt")
        else:
            print("[!] No proxies found. Running without proxy.")
    
    # Ask for number of threads
    max_threads = min(len(wallets), 10)
    while True:
        try:
            num_threads = int(input(f"\nNumber of threads? (1-{max_threads}): "))
            if 1 <= num_threads <= max_threads:
                break
            print(f"[-] Please enter a number between 1-{max_threads}")
        except ValueError:
            print("[-] Please enter a valid number")
    
    print(f"[+] Running with {num_threads} threads...")
    print("[+] Captcha not required - running without captcha solver")
    
    # Main loop
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE #{cycle} STARTING - {len(wallets)} wallets")
        print(f"{'='*60}")
        
        successful = 0
        failed = 0
        total_points = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for idx, wallet in enumerate(wallets, 1):
                proxy = proxies[(idx - 1) % len(proxies)] if proxies else None
                future = executor.submit(train_wallet_threaded, wallet, idx, len(wallets), hf_api_key, proxy)
                futures[future] = idx
            
            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    successful += 1
                    total_points += result.get("points", 0)
                else:
                    failed += 1
                    total_points += result.get("points", 0)
        
        # Summary
        print(f"\n{'─'*50}")
        print(f"📊 CYCLE #{cycle} COMPLETE")
        print(f"   ├─ Wallets: {successful}/{len(wallets)} successful")
        print(f"   ├─ Failed: {failed}")
        print(f"   ├─ Threads: {num_threads}")
        print(f"   └─ Points: +{total_points}")
        
        # Wait 1 minute before starting the next cycle
        wait_seconds = 60
        print(f"\n⏳ Waiting {wait_seconds} seconds before next cycle...")
        print(f"{'─'*50}")
        
        for remaining in range(wait_seconds, 0, -1):
            sys.stdout.write(f"\r⏰ Next cycle in: {remaining:02d} seconds   ")
            sys.stdout.flush()
            time.sleep(1)
        print("\n")


def menu_clear_missions():
    """Menu option 3: Clear all missions/tasks"""
    print("\n" + "="*60)
    print("         📋 CLEAR MISSIONS BOT")
    print("="*60)
    
    wallets = load_wallets()
    if not wallets:
        print("❌ No wallets found in wallet.json!")
        print("Please create accounts first using option 1.")
        return
    
    print(f"[+] Loaded {len(wallets)} wallets from wallet.json")
    
    # Ask about proxy
    use_proxy = input("\nDo you want to use proxy? (y/n): ").strip().lower()
    proxies = []
    if use_proxy == 'y':
        proxies = load_proxies("proxy.txt")
        if proxies:
            print(f"[+] Loaded {len(proxies)} proxies from proxy.txt")
        else:
            print("[!] No proxies found. Running without proxy.")
    
    successful = 0
    failed = 0
    total_points = 0
    
    for idx, wallet in enumerate(wallets, 1):
        print(f"\n┌─ 💼 Wallet {idx}/{len(wallets)}: {wallet['name']}")
        
        proxy = proxies[(idx - 1) % len(proxies)] if proxies else None
        if proxy:
            print(f"│  🌐 Proxy: {mask_proxy(proxy)}")
        print(f"│")
        
        bot = HumanoidTraining(
            wallet,
            fingerprint=BrowserFingerprint.generate(),
            proxy=proxy
        )
        
        if not bot.login():
            print(f"└─ ❌ Login failed\n")
            failed += 1
            continue
        
        time.sleep(1)
        
        points = bot.clear_all_tasks(show_output=True)
        if points >= 0:
            successful += 1
            total_points += points
        else:
            failed += 1
        
        print(f"└─ ✅ Done")
        
        if idx < len(wallets):
            delay = random.randint(3, 6)
            print(f"\n⏳ Next wallet in {delay}s...")
            time.sleep(delay)
    
    # Summary
    print(f"\n{'─'*50}")
    print(f"📊 CLEAR MISSIONS COMPLETE")
    print(f"   ├─ Wallets: {successful}/{len(wallets)} successful")
    print(f"   ├─ Failed: {failed}")
    print(f"   └─ Total Points: +{total_points}")
    print(f"{'─'*50}")


def main():
    """Main menu"""
    # ANSI color codes
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    banner = """
   __ __  __ __  ___ ___   ____  ____    ___  ____  ___   
  |  |  ||  |  ||   |   | /    ||    \\  /   \\|    ||   \\  
  |  |  ||  |  || _   _ ||  o  ||  _  ||     ||  | |    \\ 
  |  _  ||  |  ||  \\_/  ||     ||  |  ||  O  ||  | |  D  |
  |  |  ||  :  ||   |   ||  _  ||  |  ||     ||  | |     |
  |  |  ||     ||   |   ||  |  ||  |  ||     ||  | |     |
  |__|__| \\__,_||___|___||__|__||__|__| \\___/|____||_____|
    """
    
    while True:
        print(f"{YELLOW}{banner}{RESET}")
        print("                https://t.me/MDFKOfficial")
        print("")
        print(" " + "-"*50)
        print("")
        print("  1. 📝 Create New Accounts")
        print("  2. 🎓 Run Daily Training")
        print("  3. ❌ Exit")
        print("")
        print(" " + "-"*50)
        
        choice = input("\n Select option (1-3): ").strip()
        
        if choice == "1":
            menu_create_accounts()
        elif choice == "2":
            menu_daily_training()
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        else:
            print("[-] Invalid option. Please select 1-3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped by user.")
