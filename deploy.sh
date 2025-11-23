#!/bin/bash

echo "🚀 Deploying Deepfake Detection App"
echo "===================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git repository not initialized"
    echo "Run: git init && git add . && git commit -m 'Initial commit'"
    exit 1
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "📝 Uncommitted changes detected"
    echo "Committing all changes..."
    git add .
    git commit -m "Deployment ready: $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Code pushed to GitHub successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Go to https://share.streamlit.io/"
    echo "2. Click 'New app'"
    echo "3. Select repository: ameencaslam/deepfake-detection-project-v5"
    echo "4. Branch: main"
    echo "5. Main file: app.py"
    echo "6. Add Kaggle credentials in secrets (Advanced settings)"
    echo "7. Click Deploy!"
    echo ""
    echo "📖 Full guide: See DEPLOYMENT.md"
else
    echo ""
    echo "❌ Failed to push to GitHub"
    echo "Make sure you have:"
    echo "  - Set up GitHub remote: git remote add origin <your-repo-url>"
    echo "  - Have write access to the repository"
fi
