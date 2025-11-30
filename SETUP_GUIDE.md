# Quick Setup Guide - CheXCA

## First Time Setup (5 minutes)

### Step 1: Install Backend Dependencies

Open a terminal in the `backend` folder:

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- FastAPI (web server)
- PyTorch (AI framework)
- OpenCV & Pillow (image processing)
- And other dependencies

**Note:** If you have a GPU, you may want to install PyTorch with CUDA support for faster inference:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Frontend Dependencies

Open a terminal in the `frontend` folder:

```bash
cd frontend
npm install
```

This will install Next.js, React, and all UI dependencies.

**Note:** This step may take 2-3 minutes.

### Step 3: Verify Model Location

Ensure your trained model is at:
```
backend/models/CheXCA-Final.pth
```

✅ Already done - your model is in place!

## Running the Application

### Option 1: Quick Start (Recommended)

Double-click `start.bat` in the root folder.

This will open two command windows:
- Backend server (Python)
- Frontend application (Next.js)

Then open your browser to: **http://localhost:3000**

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open: **http://localhost:3000**

## First Run Checklist

- [ ] Backend server starts without errors
- [ ] Frontend compiles successfully
- [ ] Browser opens to localhost:3000
- [ ] You can upload an X-ray image
- [ ] Analysis completes and shows results

## Common First-Time Issues

### "Module not found" error (Backend)
- Solution: Activate virtual environment and reinstall
  ```bash
  cd backend
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```

### "Command not found: npm" (Frontend)
- Solution: Install Node.js from https://nodejs.org/
- Recommended version: 18 LTS or higher

### Model loading takes long
- Normal on first run (loading 460MB model)
- Subsequent requests are much faster

### Port already in use
- Backend: Change PORT in `backend/config.py`
- Frontend: Run with `npm run dev -- -p 3001`

## Testing the Application

1. Find a sample chest X-ray image (PNG, JPG)
2. Drag and drop it into the upload area
3. Click "Analyze X-ray"
4. Wait for results (2-5 seconds on CPU)

You should see:
- Primary diagnosis
- Probability charts
- Heatmap visualization
- Co-occurrence matrix

## Next Steps

- Read the full README.md for detailed documentation
- Customize colors in `frontend/tailwind.config.js`
- Add your own sample X-ray images for testing
- Explore the API at http://localhost:8000/docs

## Need Help?

Check the troubleshooting section in README.md for detailed solutions.

---

**Happy diagnosing!** 💙💚💜
