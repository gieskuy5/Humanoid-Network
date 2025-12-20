# 🤖 Humanoid Network Bot

Automated bot for Humanoid Network - Create accounts and run daily training tasks. Dont Forget To Join My Channel For More Updates https://t.me/+inuri73xqzYzNzll

## 📋 Features

- **Account Creation**: Automatically create new accounts with referral codes
- **Daily Training**: Submit models and datasets to earn points
- **Proxy Support**: Rotate through proxies for each account
- **Auto Retry**: Handles server errors with exponential backoff
- **Twitter Username**: Auto-generates and sets Twitter usernames

## 🚀 Quick Start

```bash
python main.py
```

## 📁 Files

| File | Description |
|------|-------------|
| `main.py` | Main script with menu (recommended) |
| `bot.py` | Account creation only |
| `daily.py` | Daily training only |
| `config.json` | API keys configuration |
| `wallet.json` | Saved wallet credentials |
| `accounts.json` | Full account data |
| `proxy.txt` | Proxy list (optional) |

## ⚙️ Configuration

### config.json
```json
{
  "HUGGINGFACE_API_KEY": "hf_your_key_here",
  "TWOCAPTCHA_API_KEY": "your_2captcha_key_here"
}
```

### proxy.txt (optional)
```
user:pass@host:port
host:port:user:pass
host:port
```

## 📖 Menu Options

1. **Create New Accounts**
   - Asks for proxy usage (y/n)
   - Number of accounts (1-100)
   - Referral code
   - Saves to `wallet.json` and `accounts.json`

2. **Run Daily Training**
   - Loads wallets from `wallet.json`
   - Submits 3 models + 3 datasets per account
   - Auto-waits until next daily reset (00:00 UTC)
   - Continuous loop with random offset

3. **Exit**

## 📦 Requirements

```bash
pip install requests eth-account
```

## 🔧 Usage Examples

### Create 10 accounts with proxy
```
Select option: 1
Use proxy? (y/n): y
How many accounts: 10
Referral code: YOUR_CODE
```

### Run daily training
```
Select option: 2
Use proxy? (y/n): y
```

## ⚠️ Notes

- Create `proxy.txt` in the same folder before using proxy feature
- First run creates `config.json` - edit with your API keys
- Daily training runs continuously until stopped (Ctrl+C)
- Wallets are saved automatically after each account creation

## 📄 License

For educational purposes only.
