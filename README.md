# CheXCA - Chest X-ray AI Diagnosis

Intelligent chest X-ray diagnosis system with explainable AI, featuring visual insights, disease co-occurrence analysis, and comprehensive medical intelligence.

## Features

- **14 Disease Classifications** - NIH ChestX-ray14 dataset pathologies
- **Explainable AI** - Grad-CAM visualizations showing what the AI focuses on
- **Interactive Dashboard** - Real-time analysis with beautiful visualizations
- **Co-occurrence Analysis** - Understanding disease correlations
- **Local Deployment** - Runs entirely on your machine

## Architecture

```
chexca-local/
├── backend/          # Python FastAPI server
│   ├── main.py                  # API endpoints
│   ├── model_inference.py       # PyTorch model loading & inference
│   ├── grad_cam.py             # Grad-CAM explainability
│   ├── config.py               # Configuration
│   ├── requirements.txt        # Python dependencies
│   └── models/
│       └── CheXCA-Final.pth    # Your trained model
└── frontend/         # Next.js application
    ├── app/
    │   ├── components/         # React components
    │   ├── lib/               # Utilities & API client
    │   ├── page.tsx           # Main page
    │   └── globals.css        # Styles
    └── package.json

```

## Prerequisites

- **Python 3.8+** - For backend
- **Node.js 18+** - For frontend
- **PyTorch** - Deep learning framework
- **GPU (Optional)** - For faster inference

## Installation

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Option 1: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```
Server will start at `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Application will open at `http://localhost:3000`

### Option 2: Quick Start Script (Windows)

Run the included batch script:
```bash
start.bat
```

This will automatically start both backend and frontend in separate windows.

## Usage

1. **Open the application** at `http://localhost:3000`
2. **Upload a chest X-ray image** (drag & drop or click to browse)
3. **Click "Analyze X-ray"** to run the AI analysis
4. **View results:**
   - Primary diagnosis with confidence score
   - Top 5 predictions chart
   - All 14 disease probabilities
   - Grad-CAM heatmap overlay
   - Disease co-occurrence matrix

## Disease Classes (NIH ChestX-ray14)

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## API Endpoints

### Health Check
```
GET /api/health
```

### Complete Analysis
```
POST /api/analyze
Body: multipart/form-data with 'file' field
Returns: predictions, heatmap, co-occurrence matrix
```

### Prediction Only
```
POST /api/predict
Body: multipart/form-data with 'file' field
Returns: disease probabilities and top predictions
```

### Explanation Only
```
POST /api/explain
Body: multipart/form-data with 'file' field
Returns: Grad-CAM visualization
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Torchvision** - Computer vision utilities
- **OpenCV** - Image processing
- **Pillow** - Image manipulation

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icons

## Troubleshooting

### Backend Issues

**Model loading error:**
- Ensure `CheXCA-Final.pth` is in `backend/models/`
- Check if the model architecture matches (DenseNet121 by default)

**CUDA not available:**
- The app will automatically fall back to CPU
- For GPU support, install PyTorch with CUDA:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

**Port 8000 already in use:**
- Change PORT in `backend/config.py`
- Update API_BASE_URL in frontend if needed

### Frontend Issues

**Cannot connect to backend:**
- Ensure backend is running at `http://localhost:8000`
- Check browser console for CORS errors
- Verify `NEXT_PUBLIC_API_URL` in frontend (defaults to localhost:8000)

**npm install fails:**
- Try deleting `node_modules` and `package-lock.json`, then run `npm install` again
- Ensure Node.js version is 18 or higher

**Port 3000 already in use:**
- Run on different port: `npm run dev -- -p 3001`

## Model Information

This application expects a PyTorch model trained on the NIH ChestX-ray14 dataset with the following characteristics:

- **Input:** 224x224 RGB images
- **Output:** 14 sigmoid probabilities (multi-label classification)
- **Architecture:** DenseNet121 (default, can be modified in `model_inference.py`)

If your model uses a different architecture, update the `_load_model()` function in `backend/model_inference.py`.

## Customization

### Changing Colors

Edit `frontend/tailwind.config.js` to modify the gradient colors:
```js
backgroundImage: {
  'gradient-medical': 'linear-gradient(135deg, #0093E9 0%, #80D0C7 50%, #A78BFA 100%)',
}
```

### Adding More Visualizations

Add new components in `frontend/app/components/` and import them in `page.tsx`.

### Modifying Model Preprocessing

Edit the `transform` in `backend/model_inference.py` to match your training preprocessing.

## Performance

- **CPU Inference:** ~2-5 seconds per image
- **GPU Inference:** ~0.5-1 second per image
- **Model Size:** ~460 MB
- **Memory Usage:** ~2-4 GB RAM

## Security & Privacy

- All processing is done **locally** on your machine
- No data is sent to external servers
- Images are temporarily stored in `backend/uploads/` during processing
- Suitable for handling sensitive medical data (ensure compliance with local regulations)

## Disclaimer

**This application is for research and educational purposes only.**

- Not intended for clinical diagnosis
- Should not replace professional medical advice
- Results should be validated by qualified healthcare professionals
- The developers assume no liability for medical decisions made using this tool

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the API documentation
3. Ensure all dependencies are correctly installed

---

**Built with medical AI research in mind** 💙💚💜
