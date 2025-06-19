# Deployment Guide - Transparent AI Chatbot

## 🚀 GitHub Deployment

### Step 1: Upload to GitHub

1. **Initialize Git Repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Transparent AI Chatbot with bias detection"
   ```

2. **Create GitHub Repository:**
   - Go to [GitHub.com](https://github.com)
   - Click "New Repository"
   - Name: `transparent-ai-chatbot`
   - Description: `AI Chatbot with Memory, Bias Detection, and Explainability`
   - Keep it Public (or Private if preferred)
   - Don't initialize with README (you already have one)

3. **Connect Local to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/transparent-ai-chatbot.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Environment Variables Setup

For deployment, you'll need to set these environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `QWEN_API_KEY` - Your Qwen/DashScope API key (optional)
- `OPENROUTER_API_KEY` - Your OpenRouter API key (optional)

## 🌐 Hosting Options

### Option 1: Railway (Recommended)
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub account
3. Click "Deploy from GitHub repo"
4. Select your `transparent-ai-chatbot` repository
5. Add environment variables in Railway dashboard
6. Railway will automatically deploy!

### Option 2: Render
1. Go to [render.com](https://render.com)
2. Connect GitHub account
3. Create "New Web Service"
4. Select your repository
5. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables
7. Deploy!

### Option 3: Heroku
1. Install Heroku CLI
2. Create `Procfile` in root directory:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
3. Commands:
   ```bash
   heroku create your-chatbot-name
   heroku config:set OPENAI_API_KEY=your_key_here
   git push heroku main
   ```

### Option 4: DigitalOcean App Platform
1. Go to DigitalOcean
2. Create new App
3. Connect GitHub repository
4. Configure build/run commands
5. Set environment variables
6. Deploy

## 🐳 Docker Deployment

If you want to use Docker:

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and Run:**
   ```bash
   docker build -t transparent-chatbot .
   docker run -p 8000:8000 transparent-chatbot
   ```

## 📊 GitHub Features to Enable

### GitHub Actions (CI/CD)
Create `.github/workflows/deploy.yml` for automatic deployment

### GitHub Pages
If you want to host documentation or a frontend

### Security Features
- Enable Dependabot for security updates
- Add branch protection rules
- Use GitHub Secrets for API keys

## 🔧 Post-Deployment Testing

After deployment, test these endpoints:
- `GET /` - API information
- `POST /chat` - Full chat with transparency
- `POST /chat/clean` - Clean formatted responses
- `POST /search` - Memory search
- `GET /stats/{session_id}` - Statistics

## 📝 Project Structure for GitHub

Your project structure is already perfect for GitHub:
```
transparent-ai-chatbot/
├── README.md
├── requirements.txt
├── main.py
├── setup.py
├── .env (excluded from git)
├── .gitignore
├── config/
├── src/
│   ├── core/
│   ├── data/
│   ├── explainability/
│   ├── fairness/
│   └── api/
├── tests/
└── examples/
```

## 🚨 Important Notes

1. **Never commit API keys** - Use environment variables
2. **Keep `.env` in `.gitignore`** - Already done
3. **Update requirements.txt** if you add new packages
4. **Test locally first** before deploying
5. **Monitor deployment logs** for any issues

## 💡 Recommended First Deployment

1. **Railway** - Easiest and most reliable
2. **Free tier available**
3. **Automatic HTTPS**
4. **Easy environment variable management**
5. **Good for prototyping and demos**

Good luck with your deployment! 🚀
