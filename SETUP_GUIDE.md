# 🚀 Quick Setup Guide - CheXCA

**Get CheXCA running in under 10 minutes!**

This guide walks you through the initial setup of the CheXCA chest X-ray analysis application. Follow these steps carefully for a smooth installation.

---

## ✅ Prerequisites Check

### Required Software

| Software | Version Required | Check Command | Download Link |
|----------|------------------|---------------|---------------|
| **Python** | 3.9.13 (Recommended) | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js** | 18.0+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | 9.0+ | `npm --version` | Included with Node.js |

### System Requirements
- **Operating System:** Windows 10/11, macOS 11+, or Ubuntu 20.04+
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 2 GB free space minimum
- **Internet:** Required for initial package downloads

### ⚠️ Python Version Important Note

**This project requires Python 3.9.13 for optimal compatibility.**

```bash
# Check your Python version
python --version
# Should output: Python 3.9.13 (or Python 3.9.x)

# If you have multiple Python versions, try:
python3.9 --version
```

If you don't have Python 3.9, download it from the official website. While Python 3.10-3.11 may work, Python 3.9 is tested and recommended for this project.

---

## 📦 First Time Setup (5-10 minutes)

### Step 1: Backend Dependencies Installation

**1.1 Navigate to backend folder:**

```bash
cd backend
```

**1.2 Create a Python virtual environment:**

This isolates project dependencies from your system Python.

```bash
# Create virtual environment named .venv
python -m venv .venv
```

**1.3 Activate the virtual environment:**

```bash
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
.venv\Scripts\activate.bat

# On macOS/Linux
source .venv/bin/activate
```

You should see `(.venv)` appear in your terminal prompt.

**1.4 Install Python packages:**

```bash
pip install -r requirements.txt
```

**Packages being installed:**
- `fastapi` - Web framework for API
- `uvicorn[standard]` - ASGI server
- `torch==2.8.0` & `torchvision` - PyTorch deep learning (CPU version)
- `timm` - ConvNeXt model backbone
- `opencv-python` - Image processing
- `Pillow` - Image handling
- `scikit-learn` - ML utilities
- `python-multipart` - File upload support

⏱️ **This step takes 3-5 minutes** depending on your internet speed.

**1.5 (Optional) GPU Support:**

If you have an NVIDIA GPU with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version with: `nvidia-smi`

---

### Step 2: Frontend Dependencies Installation

**2.1 Open a NEW terminal and navigate to frontend folder:**

```bash
cd frontend
```

**2.2 Install Node.js packages:**

```bash
npm install
```

**Packages being installed:**
- `next` - React framework (14.0.3)
- `react` & `react-dom` - UI library (18.x)
- `typescript` - Type checking
- `tailwindcss` - CSS framework
- `recharts` - Data visualization
- `axios` - API client
- `lucide-react` - Icons
- `jspdf` & `html2canvas` - PDF generation
- And various type definitions...

⏱️ **This step takes 2-3 minutes**.

---

### Step 3: Download and Install Model Files

**⚠️ IMPORTANT: Model files are NOT included in the repository due to their large size (433 MB).**

**3.1 Download the models package:**

The trained model files are distributed separately. You should receive a ZIP file containing:
```
CheXCA-Models.zip
├── chexca_state_dict.pth    (433 MB - Main model file)
└── README.txt               (Model information)
```

**Where to get the models:**
- Contact the project maintainer for the models download link
- Or use your own trained CheXCA model files

**3.2 Extract and copy model files:**

1. **Extract the ZIP file** to a temporary location
2. **Navigate to the repository's backend folder:**
   ```bash
   cd backend/models
   ```
3. **Copy all contents from the extracted ZIP to the `models` folder:**
   
   **Windows:**
   ```bash
   # Copy from Downloads (adjust path as needed)
   copy C:\Users\YourName\Downloads\CheXCA-Models\* .\
   ```
   
   **macOS/Linux:**
   ```bash
   # Copy from Downloads (adjust path as needed)
   cp ~/Downloads/CheXCA-Models/* ./
   ```
   
   **Or manually:**
   - Simply drag and drop the `chexca_state_dict.pth` file into `backend/models/` folder

**3.3 Verify model installation:**

Check that the model file exists at:
```
backend/models/chexca_state_dict.pth
```

**3.4 Verify file size:**

The model file should be approximately **433 MB** (state dict format).


```

**Troubleshooting:**
- ❌ **File not found:** Ensure you copied to the correct `backend/models/` directory
- ❌ **Wrong size:** Re-download the ZIP file, it may be corrupted
- ❌ **Permission errors:** Run terminal as administrator or use `sudo` (Linux/Mac)

✅ **Setup Complete!** You're ready to run the application.

---

## ▶️ Running the Application

### Option 1: Quick Start with Batch Script (Windows - Recommended)

**Simply double-click `start.bat` in the project root folder.**

This will:
- ✅ Open two terminal windows (Backend & Frontend)
- ✅ Activate Python virtual environment automatically
- ✅ Start both servers
- ✅ Display startup logs

Wait for both servers to show "ready" status, then open your browser to:
**http://localhost:3000**

---

### Option 2: Manual Start (Cross-Platform)

**Terminal 1 - Start Backend Server:**

```bash
# Navigate to backend
cd backend

# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

# Start server
python main.py
```

**Expected output:**
```
✓ Loaded model from state dict
✓ Model ready for inference (14 classes)
Model size: 432.99 MB
Using device: cpu
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

✅ Backend is ready when you see "Uvicorn running on http://0.0.0.0:8000"

**Terminal 2 - Start Frontend Server:**

```bash
# Navigate to frontend (new terminal)
cd frontend

# Start development server
npm run dev
```

**Expected output:**
```
▲ Next.js 14.0.3
  - Local:        http://localhost:3000
  - Environments: .env.local

✓ Ready in 2.5s
○ Compiling / ...
✓ Compiled / in 1.2s
```

✅ Frontend is ready when you see "Ready in X.Xs"

---

### Option 3: Background Processes (Linux/macOS)

```bash
# Start backend in background
cd backend && source .venv/bin/activate && python main.py &

# Start frontend in background
cd frontend && npm run dev &
```

---

## ✅ Verification Checklist

After starting both servers, verify:

- [ ] **Backend running:** Open http://localhost:8000/api/health in browser
  - Should return: `{"status": "healthy", "model_loaded": true, ...}`
  
- [ ] **Frontend running:** Open http://localhost:3000 in browser
  - Should see the CheXCA landing page with upload section
  
- [ ] **No error messages** in either terminal window
  
- [ ] **Can upload an image:** Drag & drop or click to upload
  
- [ ] **Analysis works:** Click "Analyze X-ray" button
  - Should complete in 2-5 seconds (CPU) or 0.5-1 second (GPU)
  
- [ ] **Results display:** See predictions, charts, heatmap, matrix
  
- [ ] **PDF export works:** Click "Download PDF Report" button

If all items are checked ✅, congratulations! Your setup is complete.

---

## 🐛 Common Setup Issues & Solutions

### Issue 1: "Python not found" or wrong version

**Problem:** Command `python --version` shows wrong version or command not found

**Solutions:**
```bash
# Try python3 command
python3 --version

# Or specify exact version
python3.9 --version

# Windows: Use py launcher
py -3.9 --version
```

If Python 3.9 is not installed, download from [python.org/downloads](https://www.python.org/downloads/)

---

### Issue 2: "Module not found" errors (Backend)

**Problem:** Import errors like `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
cd backend

# Ensure virtual environment is activated (you should see (.venv) in prompt)
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep fastapi  # Linux/Mac
pip list | findstr fastapi  # Windows
```

---

### Issue 3: "Command not found: npm" (Frontend)

**Problem:** npm command doesn't work

**Solution:**
1. Install Node.js from [nodejs.org](https://nodejs.org/) (LTS version recommended)
2. Restart your terminal after installation
3. Verify: `node --version` and `npm --version`
4. Both should show version numbers

---

### Issue 4: Port already in use

**Problem:** `Address already in use` or `Port 8000/3000 is in use`

**Solution for Backend (Port 8000):**
```bash
# Windows - Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Linux/Mac - Kill process
lsof -ti:8000 | xargs kill -9

# Or change port in backend/config.py
PORT = 8001  # Use different port
```

**Solution for Frontend (Port 3000):**
```bash
# Run on different port
npm run dev -- -p 3001

# Then access at http://localhost:3001
```

---

### Issue 5: Model loading fails

**Problem:** `Model file not found` or `Can't get attribute 'True_CHEXCA'`

**Solution:**
1. Verify file exists: `backend/models/chexca_state_dict.pth` (433 MB)
2. Check `MODEL_PATH` in `backend/config.py`
3. Ensure you're using state dict format, not full model
4. If you have `chexca_full_model.pth`, extract state dict:
   ```bash
   cd backend
   python extract_weights.py
   ```

---

### Issue 6: Virtual environment issues

**Problem:** Multiple virtual environments or Anaconda interference

**Solution:**
1. **Use only `.venv`** (not `venv`, `env`, etc.)
2. **Deactivate Anaconda:**
   ```bash
   conda deactivate
   ```
3. **Delete incorrect environments:**
   ```bash
   # Delete old venv (if exists)
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   ```
4. **Create fresh .venv:**
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

---

### Issue 7: GPU not detected

**Problem:** Model runs slow, startup shows "Using device: cpu"

**This is normal behavior!** GPU is optional.

**For GPU acceleration:**
1. **Verify CUDA:** `nvidia-smi` (should show GPU info)
2. **Install GPU PyTorch:**
   ```bash
   # Activate venv first
   cd backend && .venv\Scripts\Activate.ps1
   
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Restart backend** - should now show "Using device: cuda"

---

### Issue 8: Frontend build errors

**Problem:** TypeScript errors, module resolution issues, or build failures

**Solution:**
```bash
cd frontend

# Clear cache and reinstall
rm -rf node_modules package-lock.json .next  # Linux/Mac
# or manually delete these folders on Windows

# Fresh install
npm install

# Clear Next.js cache
npm run dev
```

---

### Issue 9: PDF generation fails

**Problem:** "Failed to generate PDF" error when clicking download

**Solution:**
1. Check browser console (F12) for detailed errors
2. Ensure all analysis results loaded properly
3. Try re-running the analysis
4. Verify `jspdf` and `html2canvas` are installed:
   ```bash
   cd frontend
   npm list jspdf html2canvas
   ```

---

### Issue 10: CORS errors in browser

**Problem:** "CORS policy" errors in browser console

**Solution:**
1. Ensure backend is running at http://localhost:8000
2. Check frontend is accessing correct API URL
3. Restart both servers
4. Clear browser cache

---

### Issue 11: Slow first analysis

**Problem:** First image analysis takes very long (>30 seconds)

**This is expected!** First analysis includes:
- Model loading into memory (433 MB)
- PyTorch warmup
- Image preprocessing pipeline initialization

**Subsequent analyses** should be much faster (2-5 seconds on CPU).

---

### Issue 12: Memory warnings or crashes

**Problem:** "Out of memory" or system slowdown

**Solution:**
1. Close other heavy applications
2. Ensure you have 8+ GB RAM
3. Restart backend to free memory
4. Consider adding more RAM or using smaller batch sizes

---

## 🧪 Testing Your Installation

### Quick Test (30 seconds)

1. **Health Check:**
   - Open http://localhost:8000/api/health in your browser
   - Should see JSON with `"status": "healthy"`

2. **Upload Test Image:**
   - Find any chest X-ray image (PNG or JPG)
   - Or use a sample from online medical image databases
   - Drag and drop into the upload area at http://localhost:3000

3. **Run Analysis:**
   - Click "Analyze X-ray" button
   - Wait 2-5 seconds for results

4. **Verify Output:**
   - ✅ Primary diagnosis card shows a disease name
   - ✅ Top 5 predictions chart displays
   - ✅ All 14 pathologies list appears
   - ✅ Grad-CAM heatmap shows colored overlay
   - ✅ Co-occurrence matrix displays (14×14 grid)
   - ✅ "Download PDF Report" button works

### Expected Results

For a typical chest X-ray with pneumonia:
```
Primary Diagnosis: Pneumonia (65.4%)
Top 5:
  1. Pneumonia: 65.4%
  2. Infiltration: 54.2%
  3. Consolidation: 32.1%
  4. Atelectasis: 18.7%
  5. Edema: 12.3%
```

### Sample Images

You can test with chest X-rays from:
- NIH ChestX-ray14 dataset (if you have access)
- Medical imaging teaching files
- Public domain chest X-ray images
- Your own research dataset

---

## 📚 Next Steps

Now that setup is complete:

1. **Read Full Documentation:** Check [README.md](README.md) for detailed features
2. **Explore API:** Visit http://localhost:8000/docs for interactive API docs
3. **Customize:** Modify colors, add features, adjust preprocessing
4. **Test Thoroughly:** Try various X-ray images to understand model behavior
5. **Monitor Performance:** Check inference times and memory usage

---

## 🆘 Still Having Issues?

If you're still experiencing problems:

1. **Check README.md** - Full troubleshooting section with more details
2. **Verify Versions:**
   ```bash
   python --version   # Should be 3.9.x
   node --version     # Should be 18+
   pip --version      # Verify pip is working
   npm --version      # Verify npm is working
   ```

3. **Check Logs:**
   - Backend errors appear in the Python terminal
   - Frontend errors appear in the npm terminal
   - Browser console (F12) shows client-side errors

4. **Clean Reinstall:**
   ```bash
   # Backend
   cd backend
   rm -rf .venv  # or delete manually
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # Frontend
   cd frontend
   rm -rf node_modules .next  # or delete manually
   npm install
   ```

---

## ✅ Setup Complete!

**Congratulations!** 🎉 Your CheXCA installation is ready.

You now have:
- ✅ Fully functional chest X-ray AI analysis system
- ✅ Local deployment with complete privacy
- ✅ 14-disease multi-label classification
- ✅ Explainable AI with Grad-CAM heatmaps
- ✅ Professional PDF report generation
- ✅ Production-ready FastAPI backend
- ✅ Modern Next.js 14 frontend

**Start analyzing chest X-rays at:** http://localhost:3000

---

<div align="center">

**Happy diagnosing!** 🏥💙

For advanced usage, customization, and API details, see [README.md](README.md)

</div>
