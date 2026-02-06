# 🤖 HUMANOID Network Bot

Automated bot for Humanoid Network (HAN) - Create accounts, daily training, and earn points.

## ✨ Features

- **📝 Create New Accounts** - Auto-generate wallets and register with referral code
- **🎓 Daily Training** - Submit AI models/datasets to earn points
- **🔄 Multi-threaded** - Support parallel processing for faster operations
- **🌐 Proxy Support** - Use proxies for each account
- **🔐 TLS Bypass** - Uses curl_cffi for anti-bot bypass

## 📋 Requirements

```bash
pip install eth-account curl_cffi
```

## 📁 File Structure

```
HUMANOID/
├── main.py              # Main bot script
├── config.json          # HuggingFace API key config
├── wallet.json          # Generated wallets storage
├── proxy.txt            # Proxy list (optional)
└── README.md
```

## ⚙️ Configuration

### config.json
```json
{
  "HUGGINGFACE_API_KEY": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### proxy.txt (optional)
```
user:pass@host:port
host:port:user:pass
host:port
```

## 🚀 Usage

```bash
py main.py
```

### Menu Options:
1. **Create New Accounts** - Generate new wallets and register
2. **Run Daily Training** - Submit AI models/datasets for points
3. **Exit**

## 📝 Notes

- Wallets are saved automatically to `wallet.json`
- Training submits models from HuggingFace
- Use proxies for large-scale operations

## 📞 Contact

Telegram: https://t.me/MDFKOfficial
