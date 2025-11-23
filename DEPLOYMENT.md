# Complete Deployment Guide - Deepfake Detection App

## 🚀 METHOD 1: Streamlit Community Cloud (RECOMMENDED - FREE & EASY)

### ✅ Why Streamlit Cloud?
- **100% FREE** forever
- **No credit card** required
- **1GB RAM** + **1GB Storage** 
- **Automatic HTTPS** and custom domain
- **Auto-redeploy** on GitHub push
- **Perfect for ML apps** like this

---

## 📋 DETAILED STEP-BY-STEP GUIDE

### PHASE 1: Prepare Kaggle Credentials (5 minutes)

#### Step 1.1: Get Your Kaggle API Token
1. Open browser and go to: **https://www.kaggle.com**
2. **Sign up** if you don't have an account (it's free!)
3. Click on your **profile picture** (top right) → Select **"Settings"**
4. Scroll down to the **"API"** section
5. Click **"Create New Token"** button
6. A file called `kaggle.json` will be downloaded to your computer

#### Step 1.2: Open the kaggle.json File
1. Find the downloaded `kaggle.json` file (usually in Downloads folder)
2. Open it with any text editor (Notepad, TextEdit, VS Code)
3. You'll see something like this:
   ```json
   {
     "username": "johndoe",
     "key": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
   }
   ```
4. **Copy both values** - you'll need them in Phase 3

---

### PHASE 2: Prepare Your GitHub Repository (2 minutes)

#### Step 2.1: Make Sure Code is on GitHub
Your code is already in this repository: `ameencaslam/deepfake-detection-project-v5`

If you made recent changes, push them:
```bash
# In your terminal, run these commands:
cd /workspaces/deepfake-detection-project-v5
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### Step 2.2: Verify Repository is Public
1. Go to: **https://github.com/ameencaslam/deepfake-detection-project-v5**
2. Check if you can see it without logging in
3. If it says "404" or "Private", make it public:
   - Click **Settings** tab
   - Scroll to **"Danger Zone"**
   - Click **"Change visibility"** → **"Make public"**

---

### PHASE 3: Deploy on Streamlit Cloud (10 minutes)

#### Step 3.1: Access Streamlit Cloud
1. Open new browser tab
2. Go to: **https://share.streamlit.io/**
3. Click **"Sign in with GitHub"**
4. Authorize Streamlit Cloud to access your GitHub repositories
5. You'll be redirected to Streamlit Cloud dashboard

#### Step 3.2: Create New App
1. Click the **"New app"** button (big blue button)
2. You'll see a form with 3 sections

#### Step 3.3: Configure Repository Settings
Fill in these fields:

**Repository:**
- Click the dropdown
- Find and select: `ameencaslam/deepfake-detection-project-v5`
- (If you don't see it, click "Refresh" or grant more GitHub permissions)

**Branch:**
- Select: `main`

**Main file path:**
- Type: `app.py`

#### Step 3.4: Configure App Settings (IMPORTANT!)
1. Click **"Advanced settings..."** button (bottom of form)
2. A new panel will open with multiple sections

**Python version:**
- Select: **3.11** (recommended)

**Secrets:**
This is the MOST IMPORTANT part! Click in the large text box and paste:
```toml
KAGGLE_USERNAME = "your_username_here"
KAGGLE_KEY = "your_api_key_here"
```

**Replace the values:**
- Replace `your_username_here` with the username from your `kaggle.json` file
- Replace `your_api_key_here` with the key from your `kaggle.json` file

**Example (with fake credentials):**
```toml
KAGGLE_USERNAME = "johndoe"
KAGGLE_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
```

⚠️ **IMPORTANT:** 
- Keep the quotes `"` around the values
- No spaces around the `=` sign
- Exact format as shown above

#### Step 3.5: Deploy!
1. Double-check all settings
2. Click the big **"Deploy!"** button
3. You'll be redirected to your app's page

#### Step 3.6: Wait for Deployment
You'll see a deployment log showing:

**Phase 1 (2-3 min):** Installing dependencies
```
Installing Python packages...
Installing torch, streamlit, opencv...
```

**Phase 2 (3-5 min):** Downloading models
```
Initializing models (first run)...
Downloading model weights from Kaggle...
```

**Phase 3 (1 min):** Starting app
```
Starting Streamlit app...
Your app is live!
```

**Total time:** 5-10 minutes for first deployment

---

### PHASE 4: Access Your Live App

#### Your App URL
Once deployed, your app will be available at:
```
https://<random-name>.streamlit.app
```

Example: `https://deepfake-detection-abc123.streamlit.app`

#### Custom URL (Optional)
1. Click **"Settings"** in Streamlit Cloud dashboard
2. Go to **"General"** tab
3. Edit **"App URL"** to something memorable:
   ```
   https://deepfake-detector-ameen.streamlit.app
   ```
4. Click **"Save"**

---

## 🎯 VERIFICATION CHECKLIST

After deployment, verify everything works:

### ✅ Step 1: Check Homepage
- [ ] Dark background with animated stars visible
- [ ] Hexagonal shapes floating
- [ ] Title "Deepfake🔍Detection" visible in white
- [ ] Radio buttons (Image/Video) visible with cyan border

### ✅ Step 2: Test Image Upload
- [ ] Click "Image" option
- [ ] File uploader box visible with cyan dashed border
- [ ] Upload a test image (JPG/PNG)
- [ ] See 3 prediction boxes: EfficientNet, Swin, Custom CNN
- [ ] Results show percentages and classifications

### ✅ Step 3: Test Video Upload
- [ ] Click "Video" option
- [ ] Upload a test video (MP4)
- [ ] Processing progress bar appears
- [ ] Results display for all 3 models

---

## 🔧 TROUBLESHOOTING

### Problem 1: "Initializing models" stuck forever
**Cause:** Kaggle credentials incorrect or missing

**Solution:**
1. Go to Streamlit Cloud dashboard
2. Click your app → **Settings** → **Secrets**
3. Verify KAGGLE_USERNAME and KAGGLE_KEY are correct
4. Check for typos, extra spaces, or missing quotes
5. Save and **Reboot app**

### Problem 2: App shows "Error: No module named..."
**Cause:** Missing dependency

**Solution:**
1. Check if the package is in `requirements.txt`
2. If not, add it
3. Git push the changes
4. Streamlit will auto-redeploy

### Problem 3: "Out of memory" error
**Cause:** Free tier has 1GB RAM limit

**Solutions:**
- Reduce batch sizes in video processing
- Process fewer video frames
- Upgrade to Streamlit Cloud paid tier ($20/month for 4GB RAM)

### Problem 4: Models not downloading
**Verify:**
```bash
# Check if your Kaggle dataset is public
# Go to: https://www.kaggle.com/datasets/ameencaslam/deepfake-detection-models
# Make sure it says "Public" not "Private"
```

### Problem 5: Dark theme but no content visible
**Solution:**
1. Hard refresh browser: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)
2. Clear browser cache
3. Try different browser

### Problem 6: Deployment failed with build error
**Check Logs:**
1. In Streamlit Cloud, click your app
2. Click **"Manage app"** → **"Logs"**
3. Look for specific error messages
4. Common issues:
   - Syntax errors in code
   - Invalid requirements.txt
   - Git repository access issues

---

## 🔄 REDEPLOYMENT & UPDATES

### Automatic Redeployment
Every time you push to GitHub, Streamlit automatically redeploys:
```bash
# Make changes to your code
git add .
git commit -m "Updated UI styling"
git push origin main

# Streamlit Cloud will automatically rebuild and redeploy!
```

### Manual Reboot
If app is running but behaving weird:
1. Go to Streamlit Cloud dashboard
2. Click your app
3. Click **"⋮" menu** (three dots)
4. Select **"Reboot app"**

### View Logs
To debug issues:
1. Click your app in dashboard
2. Click **"Manage app"**
3. Click **"Logs"** tab
4. See real-time logs

---

## 💰 COST BREAKDOWN

### Streamlit Cloud FREE Tier:
- **Cost:** $0/month
- **Resources:** 1GB RAM, 1GB storage
- **Apps:** Unlimited public apps
- **Traffic:** Unlimited visitors
- **Support:** Community support

### Streamlit Cloud PAID Tier:
- **Cost:** $20/month per user
- **Resources:** 4GB RAM, 10GB storage
- **Apps:** Unlimited (public + private)
- **Features:** Priority support, SSO, team collaboration

**For this app:** FREE tier is sufficient!

---

## 🌐 SHARING YOUR APP

Once deployed, share with:

### Direct Link
```
https://your-app-name.streamlit.app
```

### Embed in Website
```html
<iframe src="https://your-app-name.streamlit.app" width="100%" height="800px"></iframe>
```

### Social Media
```
🔍 Check out my AI-powered Deepfake Detector!
🎯 3 advanced models (EfficientNet, Swin, Custom CNN)
🚀 Live demo: https://your-app-name.streamlit.app
```

---

## 📊 MONITORING & ANALYTICS

### View App Statistics
1. Go to Streamlit Cloud dashboard
2. Click your app
3. See metrics:
   - Active users
   - Daily visitors
   - Resource usage
   - Error rates

### Set Up Alerts
1. Click **Settings** → **Notifications**
2. Enable email alerts for:
   - Deployment failures
   - Resource limits exceeded
   - App crashes

---

---

## 🐳 Alternative: Deploy with Docker

### Build and Run Locally
```bash
docker build -t deepfake-detection .
docker run -p 8501:8501 deepfake-detection
```

### Deploy to Cloud Platforms
- **Heroku**: Use `Dockerfile` and heroku.yml
- **Google Cloud Run**: `gcloud run deploy`
- **AWS ECS**: Push to ECR and deploy
- **Azure Container Apps**: Deploy from Docker Hub

---

## 📝 Important Notes

### Model Files
- Models are automatically downloaded on first run via `setup.sh`
- Requires Kaggle credentials in Streamlit secrets
- Alternative: Manually upload models to `runs/models/` directory

### File Size Limits
- Streamlit Cloud: 1GB total app size
- Max upload: 200MB (configured in config.toml)
- Models total: ~390MB (should fit)

### Environment Variables
Set in Streamlit Cloud secrets:
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

---

## 🔧 Troubleshooting

### Models not downloading?
1. Check Kaggle credentials in secrets
2. Verify dataset is public: `ameencaslam/deepfake-detection-models`
3. Check deployment logs for errors

### Out of memory?
1. Streamlit Cloud free tier: 1GB RAM
2. Models run on CPU (no GPU needed)
3. Reduce batch sizes if needed

### App running slow?
1. First load is slower (model initialization)
2. Consider caching with `@st.cache_resource`
3. Upgrade to Streamlit Cloud paid tier for more resources

---

## 🌐 Your App URL
After deployment, your app will be available at:
```
https://deepfake-detection-{random-id}.streamlit.app
```

You can customize the URL in Streamlit Cloud settings.
