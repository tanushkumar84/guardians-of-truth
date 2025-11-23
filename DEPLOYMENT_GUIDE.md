# Guardians of Truth - Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

### 1.1 Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Guardians of Truth Deepfake Detection"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 1.2 Verify Required Files
Ensure these files are in your repository:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ All utility files (`utils_*.py`)
- ✅ Model files in `runs/models/` (if pre-trained models exist)

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign in to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 2.2 Create New App
1. Click **"New app"** button
2. Select your repository from the dropdown
3. Choose **main** branch
4. Set main file path: `app.py`
5. Click **"Advanced settings"** (optional)

### 2.3 Configure App Settings
**App URL (Custom subdomain)**:
- Custom subdomain: `guardiansoftruth`
- Full URL will be: `guardiansoftruth.streamlit.app`

**Environment Variables** (if needed):
- Add any API keys or secrets here
- Example: `HUGGINGFACE_TOKEN`, `MLFLOW_TRACKING_URI`

### 2.4 Deploy
1. Click **"Deploy!"**
2. Wait 3-5 minutes for initial deployment
3. Monitor build logs for any errors

## Step 3: Post-Deployment

### 3.1 Test Your App
Visit: `https://guardiansoftruth.streamlit.app`

Test all features:
- ✅ Image upload and detection
- ✅ Video upload and analysis
- ✅ News article analysis
- ✅ Full frame analysis
- ✅ Report generation and download
- ✅ Navigation (Home | About Us)

### 3.2 Monitor Resources
Streamlit Cloud free tier limits:
- 1 GB RAM
- 1 CPU core
- Apps sleep after 7 days of inactivity

**Note**: Your app uses heavy models (EfficientNet, Swin, CNN). Monitor memory usage and consider:
- Using model checkpointing
- Loading models on-demand
- Upgrading to paid tier if needed

## Step 4: Update and Maintain

### 4.1 Update App
To update your deployed app:
```bash
git add .
git commit -m "Your update message"
git push origin main
```
Streamlit Cloud auto-deploys on push to main branch.

### 4.2 Manual Reboot
If app crashes or needs restart:
1. Go to your app dashboard on Streamlit Cloud
2. Click **"⋮"** menu
3. Select **"Reboot app"**

### 4.3 View Logs
- Click **"Manage app"** → **"Logs"**
- Monitor for errors or warnings
- Debug issues in real-time

## Troubleshooting

### Common Issues

**1. Build Fails - Dependencies**
```
Error: Could not find a version that satisfies the requirement
```
**Solution**: Check `requirements.txt` for version conflicts. Pin specific versions.

**2. Out of Memory**
```
Error: MemoryError or app crashes
```
**Solution**: 
- Use `@st.cache_resource` for model loading
- Implement lazy loading for heavy models
- Consider upgrading to paid tier

**3. OpenCV/MediaPipe Issues**
```
Error: ImportError: libGL.so.1
```
**Solution**: Ensure `packages.txt` includes:
```
libgl1
libglib2.0-0
```

**4. Model Files Missing**
```
Error: FileNotFoundError: runs/models/
```
**Solution**: 
- Either push model files to GitHub (if <100MB each)
- Or download models on first run using Hugging Face Hub
- Or use GitHub LFS for large files

**5. Slow Performance**
```
App is slow or times out
```
**Solution**:
- Use Streamlit session state to cache results
- Implement progress bars for long operations
- Set `server.enableCORS = false` in config

## Custom Domain (Optional)

For custom domain (e.g., `guardiansoftruth.com`):
1. Upgrade to Streamlit Cloud Pro/Teams
2. Add CNAME record in your DNS:
   - Name: `www`
   - Value: `guardiansoftruth.streamlit.app`
3. Configure in Streamlit Cloud settings

## Model Storage Options

### Option 1: GitHub Repository (Small Models <100MB)
- Push models directly to repo
- Pros: Simple, no extra setup
- Cons: GitHub file size limits

### Option 2: Hugging Face Hub (Recommended)
```python
from huggingface_hub import hf_hub_download

# In app.py, add this before model loading
model_path = hf_hub_download(
    repo_id="your-username/guardians-of-truth-models",
    filename="xception_model.pth"
)
```

### Option 3: GitHub LFS (Large Files)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model files with LFS"
```

## Security Best Practices

1. **Secrets Management**
   - Use Streamlit secrets for API keys
   - Never commit secrets to GitHub
   - Add secrets in Streamlit Cloud dashboard

2. **Input Validation**
   - Your app already validates file types
   - Ensure max upload size is set (200MB)

3. **Rate Limiting**
   - Consider adding rate limiting for production
   - Monitor usage in Streamlit Cloud analytics

## Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Your App Dashboard**: [share.streamlit.io/your-app](https://share.streamlit.io)

---

## Quick Deployment Checklist

- [ ] Code pushed to GitHub repository
- [ ] `app.py` in repository root
- [ ] `requirements.txt` with all dependencies
- [ ] `packages.txt` for system libraries
- [ ] `.streamlit/config.toml` configured
- [ ] Streamlit Cloud account created
- [ ] App deployed with subdomain `guardiansoftruth`
- [ ] All features tested on live site
- [ ] Models loaded successfully
- [ ] No memory/performance issues

**Your app will be live at**: `https://guardiansoftruth.streamlit.app`

🎉 **Congratulations! Guardians of Truth is now deployed!**
