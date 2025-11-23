# 🚀 Quick Deploy Guide - 5 Minutes to Live App!

## ⚡ FASTEST PATH TO DEPLOYMENT

### 📸 Step 1: Get Kaggle Credentials (2 min)
1. Go to: https://www.kaggle.com/settings/account
2. Click **"Create New Token"**
3. Download `kaggle.json`
4. Open it and copy the **username** and **key**

### 🔄 Step 2: Push to GitHub (30 sec)
```bash
./deploy.sh
```
Or manually:
```bash
git add .
git commit -m "Deploy"
git push origin main
```

### 🌐 Step 3: Deploy on Streamlit (2 min)
1. Go to: **https://share.streamlit.io/**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - Repository: `ameencaslam/deepfake-detection-project-v5`
   - Branch: `main`
   - File: `app.py`
5. Click **"Advanced settings"**
6. In **Secrets**, paste:
   ```toml
   KAGGLE_USERNAME = "paste_your_username_here"
   KAGGLE_KEY = "paste_your_key_here"
   ```
7. Click **"Deploy!"**

### ⏳ Step 4: Wait (5-10 min)
Watch the deployment log:
- ✅ Installing packages...
- ✅ Downloading models...
- ✅ Starting app...
- 🎉 **Your app is live!**

---

## 🎯 That's It!

Your app URL: `https://[your-app-name].streamlit.app`

**Share it with the world! 🌍**

---

## 🆘 Problems?

### Stuck on "Initializing models"?
- Check Kaggle credentials are correct
- Make sure you used exact format with quotes

### Models not downloading?
- Verify your Kaggle dataset is public
- Check you have both username AND key

### Need detailed help?
- See full guide: `DEPLOYMENT.md`
- Check troubleshooting section

---

## ✨ Bonus: Custom URL

1. In Streamlit Cloud dashboard
2. Settings → General
3. Change app URL to something cool:
   ```
   deepfake-detector-ameen
   ```

---

**Total Time: 5 minutes**
**Total Cost: $0 (FREE forever)**
**Your AI app is now LIVE! 🎉**
