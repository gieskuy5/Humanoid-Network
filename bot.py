import requests
import json
from eth_account import Account
from eth_account.messages import encode_defunct
import time
from datetime import datetime, timezone, timedelta
import random
import os
import sys


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_config():
    """Load configuration from config.json"""
    try:
        if not os.path.exists('config.json'):
            default_config = {
                "HUGGINGFACE_API_KEY": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx",
                "TWOCAPTCHA_API_KEY": "your_2captcha_api_key_here"
            }
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=2)
            print("⚠️  config.json created. Please edit with your API keys.")
            return default_config
        
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Config error: {e}")
        return {"HUGGINGFACE_API_KEY": None, "TWOCAPTCHA_API_KEY": None}


def load_proxies(filename="proxy.txt"):
    """Load proxies from file"""
    proxies = []
    try:
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r') as f:
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


def load_wallets():
    """Load wallets from wallet.json"""
    try:
        with open('wallet.json', 'r') as f:
            wallets_data = json.load(f)
        
        wallets = []
        for wallet_obj in wallets_data:
            for key, info in wallet_obj.items():
                wallets.append({'name': key, 'address': info['address'], 'private_key': info['private_key']})
        return wallets
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"❌ Error loading wallets: {e}")
        return []


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


class TwoCaptchaSolver:
    """2Captcha reCAPTCHA v2 solver"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.2captcha.com"

    def solve_recaptcha(self, website_url, website_key, user_agent=None):
        """Solve reCAPTCHA v2"""
        print(f"   ├─ 🔐 Solving captcha...", end="", flush=True)
        
        try:
            response = requests.post(f"{self.base_url}/createTask", json={
                "clientKey": self.api_key,
                "task": {
                    "type": "RecaptchaV2TaskProxyless",
                    "websiteURL": website_url,
                    "websiteKey": website_key,
                    "userAgent": user_agent
                }
            }, timeout=30)
            
            result = response.json()
            if result.get("errorId") != 0:
                print(f" ❌")
                return None
            
            task_id = result.get("taskId")
            
            for attempt in range(60):
                time.sleep(3)
                if attempt % 5 == 0: print(".", end="", flush=True)
                
                result = requests.post(f"{self.base_url}/getTaskResult", json={
                    "clientKey": self.api_key,
                    "taskId": task_id
                }, timeout=30).json()
                
                if result.get("status") == "ready":
                    print(f" ✅ ({(attempt+1)*3}s)")
                    return result.get("solution", {}).get("gRecaptchaResponse")
            
            print(f" ❌ Timeout")
            return None
        except Exception as e:
            print(f" ❌ {e}")
            return None


class BrowserFingerprint:
    """Generate browser fingerprints"""
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    ]
    
    CHROME_VERSIONS = ["129", "130", "131", "132", "143"]

    @staticmethod
    def generate():
        ua = random.choice(BrowserFingerprint.USER_AGENTS)
        platform = "Windows" if "Windows" in ua else "Macintosh" if "Macintosh" in ua else "Linux"
        browser = "Microsoft Edge" if "Edge" in ua else "Chromium"
        version = random.choice(BrowserFingerprint.CHROME_VERSIONS)
        
        return {
            "user_agent": ua,
            "sec_ch_ua": f'"{browser}";v="{version}", "Chromium";v="{version}", "Not A(Brand";v="24"',
            "sec_ch_ua_mobile": "?0",
            "sec_ch_ua_platform": f'"{platform}"',
            "accept_language": random.choice(["en-US,en;q=0.9", "en-US,en;q=0.9,id;q=0.8", "en-GB,en;q=0.9"])
        }


# ============================================================
# ACCOUNT CREATION BOT
# ============================================================

class HumanoidNetwork:
    """Humanoid Network Account Creator"""
    def __init__(self, fingerprint=None, proxy=None):
        self.base_url = "https://prelaunch.humanoidnetwork.org/api"
        self.session = requests.Session()
        self.wallet = None
        self.account = None
        self.token = None
        self.user_data = None
        self.fingerprint = fingerprint or BrowserFingerprint.generate()
        
        if proxy:
            parsed_proxy = parse_proxy(proxy)
            if parsed_proxy:
                self.session.proxies = parsed_proxy

    def get_headers(self, include_auth=False):
        """Generate headers with fingerprint"""
        headers = {
            "accept": "*/*",
            "accept-language": self.fingerprint["accept_language"],
            "cache-control": "no-cache",
            "content-type": "application/json",
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

    def save_account_data(self, filename="accounts.json", wallet_number=1):
        """Save complete account data to file"""
        print("[*] Saving account data...")
        user_info = self.get_user_info()
        
        account_data = {
            "timestamp": datetime.now().isoformat(),
            "wallet": self.wallet,
            "token": self.token,
            "user_data": user_info,
            "referral_code": self.user_data.get('referralCode', 'N/A'),
            "fingerprint": self.fingerprint
        }
        
        try:
            with open(filename, 'r') as f:
                accounts = json.load(f)
        except FileNotFoundError:
            accounts = []
        
        accounts.append(account_data)
        
        with open(filename, 'w') as f:
            json.dump(accounts, f, indent=2)
        
        print(f"[+] Account data saved to {filename}")
        self.save_wallet_only(wallet_number)
        return account_data

    def save_wallet_only(self, wallet_number):
        """Save wallet to wallet.json"""
        wallet_filename = "wallet.json"
        
        try:
            with open(wallet_filename, 'r') as f:
                wallets = json.load(f)
        except FileNotFoundError:
            wallets = []
        
        wallet_entry = {
            f"wallet{wallet_number}": {
                "address": self.wallet['address'],
                "private_key": self.wallet['private_key']
            }
        }
        
        wallets.append(wallet_entry)
        
        with open(wallet_filename, 'w') as f:
            json.dump(wallets, f, indent=2)
        
        print(f"[+] Wallet saved to {wallet_filename}")


# ============================================================
# TRAINING BOT
# ============================================================

class HumanoidTraining:
    """Humanoid Network Training Bot"""
    RECAPTCHA_SITEKEY = "6LcdlCcsAAAAAJGvjt5J030ySi7htRzB6rEeBgcP"
    WEBSITE_URL = "https://prelaunch.humanoidnetwork.org/training"
    
    MODELS_POOL = [
        "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5",
        "black-forest-labs/FLUX.1-dev", "meta-llama/Llama-3.2-1B", "mistralai/Mistral-7B-v0.1",
        "google/gemma-2b", "microsoft/phi-2", "openai/whisper-large-v3",
        "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"
    ]
    
    DATASETS_POOL = [
        "HuggingFaceFW/fineweb-edu", "allenai/c4", "bigcode/the-stack-v2",
        "teknium/OpenHermes-2.5", "Open-Orca/OpenOrca", "wikimedia/wikipedia",
        "EleutherAI/pile", "databricks/databricks-dolly-15k", "tatsu-lab/alpaca"
    ]

    def __init__(self, wallet_data, fingerprint=None, hf_api_key=None, captcha_api_key=None, proxy=None):
        self.base_url = "https://prelaunch.humanoidnetwork.org/api"
        self.session = requests.Session()
        self.wallet = wallet_data
        self.fingerprint = fingerprint or BrowserFingerprint.generate()
        self.hf_api_key = hf_api_key
        self.captcha_solver = TwoCaptchaSolver(captcha_api_key) if captcha_api_key else None
        self.account = Account.from_key(self.wallet['private_key'])
        self.token = None
        self.user_data = None
        self.submitted_items = set()
        self.total_points = 0
        
        if proxy:
            parsed_proxy = parse_proxy(proxy)
            if parsed_proxy:
                self.session.proxies = parsed_proxy

    def get_headers(self, include_auth=False):
        """Generate headers"""
        headers = {
            "accept": "*/*",
            "accept-language": self.fingerprint["accept_language"],
            "content-type": "application/json",
            "origin": "https://prelaunch.humanoidnetwork.org",
            "referer": "https://prelaunch.humanoidnetwork.org/training",
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

    def login(self):
        """Login using wallet"""
        addr_short = f"{self.wallet['address'][:8]}...{self.wallet['address'][-6:]}"
        print(f"   ├─ 🔐 Logging in: {addr_short}")
        
        try:
            resp = self.session.post(f"{self.base_url}/auth/nonce", 
                json={"walletAddress": self.wallet['address']}, headers=self.get_headers())
            if resp.status_code != 200: 
                print(f"   ├─ ❌ Failed to get nonce: {resp.status_code}")
                return False
            nonce_data = resp.json()
            
            signature = self.sign_message(nonce_data['message'])
            if not signature: 
                print(f"   ├─ ❌ Failed to sign message")
                return False
            
            resp = self.session.post(f"{self.base_url}/auth/authenticate", json={
                "walletAddress": self.wallet['address'],
                "signature": signature,
                "message": nonce_data['message']
            }, headers=self.get_headers())
            
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
            print(f"   ├─ ❌ Login error: {e}")
            return False

    def _auto_twitter(self):
        """Automatically check and set Twitter username"""
        if self.user_data and self.user_data.get('twitterId'):
            print(f"   ├─ 🐦 Twitter: @{self.user_data['twitterId']} (exists)")
            return
        
        username = TwitterUsernameGenerator.generate()
        try:
            resp = self.session.post(f"{self.base_url}/user/update-x-username",
                json={"twitterUsername": username}, headers=self.get_headers(include_auth=True))
            if resp.status_code == 200 and resp.json().get('success'):
                print(f"   ├─ 🐦 Twitter: @{username} (new)")
                if self.user_data: 
                    self.user_data['twitterId'] = username
            else:
                print(f"   ├─ 🐦 Twitter: ❌ Failed to set")
        except Exception as e:
            print(f"   ├─ 🐦 Twitter: ❌ Error: {e}")

    def _load_submitted(self):
        """Load already submitted items"""
        try:
            resp = self.session.get(f"{self.base_url}/training", headers=self.get_headers(include_auth=True))
            if resp.status_code == 200:
                trainings = resp.json()
                for t in trainings:
                    self.submitted_items.add(t['fileName'])
                    self.submitted_items.add(t['fileUrl'])
                if trainings:
                    models = sum(1 for t in trainings if t['fileType'] == 'model')
                    datasets = len(trainings) - models
                    print(f"   ├─ 📋 Previous: {models} models, {datasets} datasets")
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
        """Get random model or dataset"""
        pool = self.MODELS_POOL if item_type == "model" else self.DATASETS_POOL
        prefix = "https://huggingface.co/" if item_type == "model" else "https://huggingface.co/datasets/"
        
        available = [i for i in pool if i not in self.submitted_items]
        if not available: available = pool
        
        name = random.choice(available)
        return {"fileName": name, "fileUrl": f"{prefix}{name}", "fileType": item_type}

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
        if item_data['fileName'] in self.submitted_items:
            return None
        
        token = None
        if self.captcha_solver:
            token = self.captcha_solver.solve_recaptcha(
                self.WEBSITE_URL, self.RECAPTCHA_SITEKEY, self.fingerprint["user_agent"])
            if not token: return None
        
        payload = {
            "fileName": item_data['fileName'],
            "fileUrl": item_data['fileUrl'],
            "fileType": item_data['fileType'],
            "description": description
        }
        if token: payload["recaptchaToken"] = token
        
        try:
            resp = self.session.post(f"{self.base_url}/training", json=payload, 
                headers=self.get_headers(include_auth=True))
            
            if resp.status_code == 200:
                data = resp.json()
                self.total_points += data['points']
                self.submitted_items.add(item_data['fileName'])
                self.submitted_items.add(item_data['fileUrl'])
                return data
            else:
                error = resp.json() if resp.text else {}
                print(f"   │     └─ ❌ {error.get('error', resp.status_code)}")
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
        
        print(f"   ├─ 📊 Daily: Models {daily['models']['completed']}/{daily['models']['limit']} | Datasets {daily['datasets']['completed']}/{daily['datasets']['limit']}")
        
        actual_models = min(model_count, models_remaining)
        actual_datasets = min(dataset_count, datasets_remaining)
        total = actual_models + actual_datasets
        
        if total == 0:
            print(f"   └─ ⚠️  Daily limits reached!")
            return True
        
        successful = 0
        
        for i in range(actual_models):
            item = self.get_random_item("model")
            name = item['fileName'][:35] + "..." if len(item['fileName']) > 35 else item['fileName']
            print(f"   ├─ 🤖 [{i+1}/{actual_models}] {name}")
            
            desc = self.get_description(item['fileName'], "model")
            result = self.submit_item(item, desc)
            
            if result:
                print(f"   │     └─ ✅ +{result['points']} pts")
                successful += 1
            
            if i < actual_models - 1 or actual_datasets > 0:
                time.sleep(random.randint(3, 6))
        
        for i in range(actual_datasets):
            item = self.get_random_item("dataset")
            name = item['fileName'][:35] + "..." if len(item['fileName']) > 35 else item['fileName']
            print(f"   ├─ 📚 [{i+1}/{actual_datasets}] {name}")
            
            desc = self.get_description(item['fileName'], "dataset")
            result = self.submit_item(item, desc)
            
            if result:
                print(f"   │     └─ ✅ +{result['points']} pts")
                successful += 1
            
            if i < actual_datasets - 1:
                time.sleep(random.randint(3, 6))
        
        print(f"   └─ 📈 Session: {successful}/{total} | +{self.total_points} pts")
        return True


# ============================================================
# MENU FUNCTIONS
# ============================================================

def create_account(referral_code, account_number, total_accounts, proxy=None):
    """Create single account with unique fingerprint and optional proxy"""
    print("\n" + "="*60)
    print(f"Creating Account {account_number}/{total_accounts}")
    print("="*60)
    
    fingerprint = BrowserFingerprint.generate()
    print(f"[+] Fingerprint: {fingerprint['user_agent'][:66]}...")
    print(f"[+] Proxy: {mask_proxy(proxy)}")
    
    bot = HumanoidNetwork(fingerprint=fingerprint, proxy=proxy)
    
    bot.generate_wallet()
    
    nonce_data = bot.get_nonce()
    if not nonce_data:
        print("[-] Failed to get nonce. Skipping account...")
        return None
    
    auth_data = bot.authenticate(nonce_data, referral_code)
    if not auth_data:
        print("[-] Authentication failed. Skipping account...")
        return None
    
    bot.get_user_info()
    bot.clear_tasks()
    account_data = bot.save_account_data(wallet_number=account_number)
    
    print("\n" + "="*60)
    print(f"ACCOUNT {account_number} COMPLETED!")
    print("="*60)
    print(f"Wallet Address: {bot.wallet['address']}")
    print(f"Private Key: {bot.wallet['private_key']}")
    print(f"Referral Code: {bot.user_data.get('referralCode', 'N/A')}")
    print(f"Total Points: {account_data['user_data']['totalPoints']}")
    print("="*60)
    
    return account_data


def menu_create_accounts():
    """Menu option 1: Create new accounts"""
    print("\n" + "="*60)
    print("         📝 CREATE NEW ACCOUNTS")
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
            num_accounts = int(input("\nHow many accounts to create? (1-100): "))
            if 1 <= num_accounts <= 100:
                break
            print("[-] Please enter a number between 1-100")
        except ValueError:
            print("[-] Please enter a valid number")
    
    # Ask for referral code
    referral_code = input("Enter referral code: ").strip()
    if not referral_code:
        referral_code = "WYPUMM"
    print(f"[+] Using referral code: {referral_code}")
    
    # Create accounts
    successful_accounts = []
    failed_accounts = 0
    
    for i in range(1, num_accounts + 1):
        proxy = proxies[(i - 1) % len(proxies)] if proxies else None
        account_data = create_account(referral_code, i, num_accounts, proxy=proxy)
        
        if account_data:
            successful_accounts.append(account_data)
        else:
            failed_accounts += 1
        
        if i < num_accounts:
            delay = random.randint(5, 10)
            print(f"\n[*] Waiting {delay} seconds before next account...")
            time.sleep(delay)
    
    # Summary
    print("\n\n" + "="*60)
    print("ALL ACCOUNTS CREATION COMPLETED!")
    print("="*60)
    print(f"Total Requested: {num_accounts}")
    print(f"Successfully Created: {len(successful_accounts)}")
    print(f"Failed: {failed_accounts}")
    print(f"\nData saved to: accounts.json, wallet.json")
    print("="*60)


def menu_daily_training():
    """Menu option 2: Run daily training"""
    print("\n" + "="*60)
    print("         🎓 DAILY TRAINING BOT")
    print("="*60)
    
    config = load_config()
    
    hf_api_key = config.get("HUGGINGFACE_API_KEY")
    if not hf_api_key or hf_api_key == "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx":
        hf_api_key = None
    
    captcha_api_key = config.get("TWOCAPTCHA_API_KEY")
    if not captcha_api_key or captcha_api_key == "your_2captcha_api_key_here":
        captcha_api_key = None
    
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
    
    # Main loop
    cycle = 0
    while True:
        cycle += 1
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
                hf_api_key=hf_api_key,
                captcha_api_key=captcha_api_key,
                proxy=proxy
            )
            
            if not bot.login():
                print(f"└─ ❌ Login failed\n")
                failed += 1
                continue
            
            time.sleep(1)
            
            if bot.do_training(model_count=3, dataset_count=3):
                successful += 1
                total_points += bot.total_points
            else:
                failed += 1
            
            if idx < len(wallets):
                delay = random.randint(5, 10)
                print(f"\n⏳ Next wallet in {delay}s...")
                time.sleep(delay)
        
        # Summary
        print(f"\n{'─'*50}")
        print(f"📊 CYCLE #{cycle} COMPLETE")
        print(f"   ├─ Wallets: {successful}/{len(wallets)} successful")
        print(f"   ├─ Failed: {failed}")
        print(f"   └─ Points: +{total_points}")
        
        # Wait for next daily reset
        next_reset = get_next_daily_reset()
        random_offset = random.randint(5 * 60, 30 * 60)
        next_run = next_reset + timedelta(seconds=random_offset)
        
        print(f"\n⏰ Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'─'*50}")
        
        wait_with_countdown(next_run, "Until next daily")


def main():
    """Main menu"""
    while True:
        print("\n" + "="*60)
        print("         🤖 HUMANOID NETWORK BOT")
        print("="*60)
        print("\n  1. 📝 Create New Accounts")
        print("  2. 🎓 Run Daily Training")
        print("  3. ❌ Exit")
        print("\n" + "="*60)
        
        choice = input("\nSelect option (1-3): ").strip()
        
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
