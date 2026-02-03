# CheXCA - Chest X-ray AI Diagnosis

**Advanced chest X-ray diagnosis system powered by deep learning with explainable AI**

CheXCA (Chest X-ray Classification & Analysis) is a production-ready medical AI application that provides intelligent chest X-ray analysis with visual explanations, disease co-occurrence patterns, and comprehensive diagnostic insights. Built with True_CHEXCA architecture (ConvNeXt-Base + CTCA + GAT Fusion) and designed for local deployment.

## Features

- **14 Disease Classifications** - Complete NIH ChestX-ray14 dataset pathologies with multi-label classification
- **Explainable AI** - Grad-CAM++ heatmap visualizations showing model attention regions
- **Interactive Dashboard** - Real-time analysis with professional data visualizations
- **Co-occurrence Analysis** - Disease correlation matrix with visual intensity mapping
- **PDF Export** - Comprehensive reports with all analysis results, charts, and visualizations
- **Local Deployment** - Complete privacy - all processing happens on your machine
- **Production Ready** - FastAPI backend + Next.js frontend with TypeScript

## Architecture

### System Overview
```
chexca-local/
├── backend/                     # Python FastAPI backend server
│   ├── main.py                 # API endpoints with lifespan management
│   ├── model_inference.py      # Model loading & inference logic
│   ├── chexca_model.py         # True_CHEXCA architecture (ConvNeXt + CTCA + GAT)
│   ├── grad_cam.py            # Grad-CAM++ explainability for ConvNeXt
│   ├── config.py              # Configuration & disease classes
│   ├── requirements.txt       # Python dependencies
│   ├── extract_weights.py     # Utility to extract state dict from full model
│   └── models/
│       └── chexca_state_dict.pth    # Trained model weights (433 MB)
└── frontend/                   # Next.js 14 application
    ├── app/
    │   ├── components/        # React components
    │   │   ├── ResultsPanel.tsx        # Main results display with PDF export
    │   │   ├── CoOccurrenceMatrix.tsx  # Disease correlation visualization
    │   │   ├── HeatmapViewer.tsx       # Grad-CAM heatmap display
    │   │   ├── UploadSection.tsx       # File upload interface
    │   │   └── HeroSection.tsx         # Landing section
    │   ├── lib/
    │   │   ├── api.ts                  # Backend API client
    │   │   ├── pdfExport.ts            # PDF generation with jsPDF
    │   │   └── utils.ts                # Utility functions
    │   ├── page.tsx           # Main application page
    │   └── globals.css        # Global styles & Tailwind config
    ├── package.json
    └── next.config.js
```

### Model Architecture: True_CHEXCA

**ConvNeXt-Base Backbone** (ImageNet-22K pretrained)
- Feature extraction from 224×224 RGB chest X-rays
- Outputs 1024-dimensional feature vectors

**CTCA (Class Token Cross-Attention)**
- Hybrid attention mechanism combining:
  - Spatial attention maps (7×7 grid)
  - Class-specific learnable tokens (14 tokens for 14 diseases)
  - Cross-attention between class tokens and spatial features

**GAT Fusion (Graph Attention Network)**
- Models disease relationships and co-occurrence patterns
- Multi-head attention (4 heads) for learning disease interactions
- Falls back to MLP if PyTorch Geometric not installed

**Per-Class Classifiers**
- Individual binary classifiers for each of the 14 diseases
- Enables multi-label prediction (multiple diseases can co-exist)

**Training Details**
- Loss: Focal Loss (γ=2.0) for handling class imbalance
- Optimizer: AdamW (lr=1e-4, weight decay=1e-4)
- EMA: Exponential Moving Average (decay=0.999) for stable weights
- Mixed Precision: Automatic Mixed Precision (AMP) for faster training
- Preprocessing: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Prerequisites

- **Python 3.9.13** - Backend runtime (critical for PyTorch compatibility)
- **Node.js 18+** - Frontend build system
- **8GB RAM minimum** - For model inference (16GB recommended)
- **GPU (Optional)** - CUDA-compatible GPU for faster inference (2-5x speedup)

> **Important:** This project requires Python 3.9.13. While Python 3.10-3.11 may work, we strongly recommend Python 3.9 for optimal compatibility with PyTorch 2.8.0 and all dependencies.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Quad-core | 8+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 5 GB |
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 13, Ubuntu 22.04 |

## Installation

### Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Create and activate a virtual environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `fastapi` - Modern web framework for building APIs
- `uvicorn[standard]` - ASGI server for production
- `torch` & `torchvision` - PyTorch deep learning framework (2.8.0)
- `timm` - PyTorch Image Models for ConvNeXt backbone
- `opencv-python` - Computer vision and image processing
- `Pillow` - Image manipulation
- `scikit-learn` - Machine learning utilities
- `python-multipart` - File upload handling

**Optional GPU Support:**
```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

**Dependencies installed:**
- `next` - React framework with server-side rendering
- `react` & `react-dom` - UI library
- `typescript` - Type safety
- `tailwindcss` - Utility-first CSS framework
- `recharts` - Charting library for data visualization
- `axios` - HTTP client for API requests
- `lucide-react` - Icon library
- `jspdf` & `html2canvas` - PDF report generation

### Model Files Setup

**CRITICAL STEP: Model files are NOT included in this repository.**

The trained model files (433 MB) are distributed separately due to GitHub file size limitations.

**To get the models:**

1. **Download the models package** - Obtain `CheXCA-Models.zip` from the project maintainer
2. **Extract the ZIP file** containing `chexca_state_dict.pth`
3. **Copy to the models folder:**
   ```bash
   # Ensure you're in the repository root
   cd backend/models
   
   # Copy the extracted model file here
   # Windows: copy C:\Path\To\chexca_state_dict.pth .\
   # macOS/Linux: cp ~/Path/To/chexca_state_dict.pth ./
   ```
4. **Verify the file:** Should be ~433 MB at `backend/models/chexca_state_dict.pth`

**See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed model installation instructions.**

## Running the Application

### Option 1: Quick Start Script (Windows - Recommended)

Double-click `start.bat` in the root directory. This will:
- Start the backend server in one terminal (port 8000)
- Start the frontend development server in another terminal (port 3000)
- Open your browser automatically to http://localhost:3000

### Option 2: Manual Start (Cross-Platform)

**Terminal 1 - Start Backend:**
```bash
cd backend
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
python main.py
```
Backend running at `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```
Frontend running at `http://localhost:3000`

### Verifying Startup

**Backend logs should show:**
```
✓ Loaded model from state dict
✓ Model ready for inference (14 classes)
Model size: 432.99 MB
Using device: cpu
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Frontend logs should show:**
```
✓ Ready in 2.5s
○ Local:   http://localhost:3000
```

## Usage Guide

### Step-by-Step Analysis

1. **Open the application** 
   - Navigate to `http://localhost:3000` in your web browser
   - Chrome, Firefox, Edge, or Safari recommended

2. **Upload a chest X-ray image**
   - Drag and drop an image into the upload area, or
   - Click the upload area to browse your files
   - Supported formats: PNG, JPG, JPEG
   - Recommended size: 224×224 to 1024×1024 pixels

3. **Run AI analysis**
   - Click the "Analyze X-ray" button
   - Wait 2-5 seconds (CPU) or 0.5-1 second (GPU)
   - Progress indicator shows analysis status

4. **Review comprehensive results:**

   **Primary Diagnosis Card**
   - Shows the most likely disease with confidence percentage
   - Color-coded confidence bar (red >70%, orange >50%, yellow >30%, blue <30%)

   **Top 5 Predictions Chart**
   - Interactive bar chart with probability percentages
   - Hover for detailed values
   - Color-coded by confidence level

   **All 14 Pathologies Grid**
   - Complete probability breakdown for all diseases
   - Progress bars for visual comparison
   - Sorted by probability (highest to lowest)

   **Grad-CAM Heatmap Visualization**
   - Side-by-side comparison of original X-ray and attention heatmap
   - Red/warm colors indicate regions the AI focused on
   - Helps understand model decision-making

   **Disease Co-Occurrence Matrix**
   - 14×14 heatmap showing disease correlation patterns
   - Darker purple indicates stronger co-occurrence
   - Interactive tooltips with exact correlation values

5. **Export PDF Report**
   - Click "Download PDF Report" button
   - Comprehensive PDF includes:
     - Primary diagnosis with confidence
     - All disease probabilities
     - Top 5 predictions chart representation
     - Original and heatmap images
     - Co-occurrence matrix visualization
     - Timestamp and disclaimer
   - Professional format suitable for presentations or documentation

## Disease Classes (NIH ChestX-ray14)

The model is trained to detect 14 thoracic pathologies from the NIH ChestX-ray14 dataset:

| # | Disease | Description |
|---|---------|-------------|
| 1 | **Atelectasis** | Lung collapse or closure |
| 2 | **Cardiomegaly** | Enlarged heart |
| 3 | **Effusion** | Fluid accumulation in pleural space |
| 4 | **Infiltration** | Substance denser than air in lung parenchyma |
| 5 | **Mass** | Pulmonary mass or lesion |
| 6 | **Nodule** | Small rounded opacity <3cm |
| 7 | **Pneumonia** | Lung infection and inflammation |
| 8 | **Pneumothorax** | Air in pleural space (collapsed lung) |
| 9 | **Consolidation** | Alveolar air replaced by fluid/tissue |
| 10 | **Edema** | Fluid accumulation in lungs |
| 11 | **Emphysema** | Alveolar damage and air trapping |
| 12 | **Fibrosis** | Lung scarring and thickening |
| 13 | **Pleural Thickening** | Thickened pleural membrane |
| 14 | **Hernia** | Organ displacement (typically hiatal) |

**Multi-Label Classification:** The model can detect multiple diseases simultaneously (e.g., a patient may have both Pneumonia and Infiltration).

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "chexca_state_dict.pth",
  "num_classes": 14,
  "device": "cpu"
}
```

#### 2. Complete Analysis (Recommended)
```http
POST /api/analyze
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: Image file (PNG, JPG, JPEG)

**Response:**
```json
{
  "predictions": {
    "Pneumonia": 0.8234,
    "Infiltration": 0.6721,
    "Atelectasis": 0.2341,
    ...
  },
  "top_predictions": [
    {"disease": "Pneumonia", "probability": 0.8234},
    {"disease": "Infiltration", "probability": 0.6721},
    ...
  ],
  "top_disease": "Pneumonia",
  "top_disease_probability": 0.8234,
  "heatmap_overlay": "data:image/png;base64,...",
  "original_image": "data:image/png;base64,...",
  "co_occurrence": [[1.0, 0.45, ...], ...],
  "disease_classes": ["Atelectasis", "Cardiomegaly", ...]
}
```

#### 3. Prediction Only
```http
POST /api/predict
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: Image file

**Response:**
```json
{
  "predictions": {...},
  "top_predictions": [...],
  "top_disease": "Pneumonia",
  "top_disease_probability": 0.8234
}
```

#### 4. Explanation Only (Grad-CAM)
```http
POST /api/explain
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: Image file

**Response:**
```json
{
  "heatmap_overlay": "data:image/png;base64,...",
  "original_image": "data:image/png;base64,..."
}
```

### Interactive API Docs

FastAPI automatically generates interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Tech Stack

### Backend Technologies
- **FastAPI 0.104+** - Modern Python web framework with async support
- **PyTorch 2.8.0** - Deep learning framework for model inference
- **Torchvision** - Computer vision utilities and transforms
- **timm (PyTorch Image Models)** - ConvNeXt backbone implementation
- **OpenCV (cv2)** - Image processing for Grad-CAM visualization
- **Pillow (PIL)** - Image manipulation and format conversion
- **scikit-learn** - Utilities for co-occurrence matrix computation
- **Uvicorn** - Lightning-fast ASGI server
- **Python 3.9.13** - Runtime environment

### Frontend Technologies
- **Next.js 14.0.3** - React framework with App Router
- **React 18** - UI library with modern hooks
- **TypeScript 5** - Static type checking
- **Tailwind CSS 3** - Utility-first CSS framework
- **Recharts 2.10+** - Composable charting library for React
- **Axios** - Promise-based HTTP client
- **Lucide React** - Beautiful open-source icons
- **jsPDF 2.5.1** - PDF generation library
- **html2canvas 1.4.1** - HTML to canvas screenshot utility
- **Node.js 18+** - JavaScript runtime

### Development Tools
- **ESLint** - Code linting and formatting
- **PostCSS** - CSS transformations
- **Python virtual environments** - Dependency isolation

## Troubleshooting

### Backend Issues

**❌ "Model file not found" error**
- **Solution:** Ensure `chexca_state_dict.pth` exists in `backend/models/`
- Check `MODEL_PATH` in `backend/config.py`
- Model should be 433 MB (state dict format)

**❌ "Can't get attribute 'True_CHEXCA'" error**
- **Cause:** PyTorch 2.6+ requires `weights_only=False` for full model objects
- **Solution:** Use the extracted state dict (`chexca_state_dict.pth`) instead of full model
- Already configured correctly in `model_inference.py`

**❌ CUDA/GPU not available**
- **Behavior:** App automatically falls back to CPU (slower but works)
- **For GPU support:** 
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- Verify GPU detection: Check backend startup logs for "Using device: cuda"

**❌ Port 8000 already in use**
- **Solution 1:** Kill the process using port 8000
  ```bash
  # Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  
  # Linux/Mac
  lsof -ti:8000 | xargs kill -9
  ```
- **Solution 2:** Change port in `backend/config.py` and update frontend API URL

**❌ "Module not found" errors**
- **Solution:** Activate virtual environment and reinstall dependencies
  ```bash
  cd backend
  .venv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```

**❌ Virtual environment confusion (multiple Python versions)**
- **Problem:** Having both `.venv` (3.9.13) and `venv` (3.13.1)
- **Solution:** Always use `.venv` with Python 3.9.13
- Delete incorrect `venv` folder if it exists

### Frontend Issues

**❌ "Cannot connect to backend" error**
- **Check 1:** Ensure backend is running at http://localhost:8000
- **Check 2:** Open http://localhost:8000/api/health in browser to verify
- **Check 3:** Check browser console for CORS or network errors
- **Check 4:** Verify `NEXT_PUBLIC_API_URL` environment variable (default: localhost:8000)

**❌ `npm install` fails**
- **Solution:** Clear cache and reinstall
  ```bash
  cd frontend
  rm -rf node_modules package-lock.json  # or delete manually on Windows
  npm install
  ```
- Ensure Node.js version is 18 or higher: `node --version`

**❌ Port 3000 already in use**
- **Solution:** Run on different port
  ```bash
  npm run dev -- -p 3001
  ```

**❌ "Module not found" or import errors**
- **Solution:** Clear Next.js cache and rebuild
  ```bash
  rm -rf .next  # or delete .next folder manually
  npm run dev
  ```

**❌ PDF generation fails or produces empty PDF**
- **Cause:** Missing analysis result data or image encoding issues
- **Solution:** Check browser console for detailed errors
- Ensure all result fields are populated before clicking download

### Performance Issues

**⚠️ Slow inference (>10 seconds per image)**
- **On CPU:** Normal behavior (2-5 seconds typical)
- **Solutions:**
  - Install GPU-accelerated PyTorch (2-5x faster)
  - Reduce image size before upload
  - Close other heavy applications

**⚠️ High memory usage**
- **Normal:** 2-4 GB RAM (model is 433 MB, requires overhead)
- **If excessive (>8 GB):** 
  - Restart backend server
  - Check for memory leaks (shouldn't happen in production)

### Common Warnings (Safe to Ignore)

✓ "torch_geometric not available. Falling back to MLP-based GAT"
- Not an error - app uses MLP instead of full GAT
- For full GAT: `pip install torch-geometric` (optional)

✓ "UserWarning: resource_tracker: There appear to be X leaked semaphore objects"
- Known PyTorch/multiprocessing warning on Windows
- Does not affect functionality

## Model Performance & Specifications

### Model Details
- **Architecture:** True_CHEXCA (ConvNeXt-Base + CTCA + GAT Fusion)
- **Input Size:** 224×224 RGB images
- **Output:** 14 sigmoid probabilities (multi-label classification)
- **Model Size:** 433 MB (state dict)
- **Parameters:** ~88M+ parameters (ConvNeXt-Base backbone)
- **Preprocessing:** ImageNet normalization

### Inference Performance

| Hardware | Avg. Time per Image | Throughput |
|----------|---------------------|------------|
| CPU (Intel i7) | 2-5 seconds | 12-30 images/min |
| GPU (RTX 3060) | 0.5-1 second | 60-120 images/min |
| GPU (RTX 4090) | 0.2-0.4 seconds | 150-300 images/min |

### Memory Requirements
- **Backend (Model + Runtime):** 2-4 GB RAM
- **Frontend (Next.js):** 200-500 MB RAM
- **Total System:** 4-8 GB RAM recommended

### Dataset Information
- **Training Data:** NIH ChestX-ray14 (112,120 frontal-view X-ray images)
- **Patients:** 30,805 unique patients
- **Labels:** 14 thoracic disease labels (multi-label)
- **Image Source:** PACS at NIH Clinical Center

## Customization & Extension

### Changing Model

To use a different trained model:

1. **Place your model file** in `backend/models/`
2. **Update `config.py`:**
   ```python
   MODEL_PATH = BASE_DIR / "models" / "your_model.pth"
   ```
3. **If architecture differs,** modify `chexca_model.py` to match your model
4. **Restart backend** and test with a sample image

### Extracting State Dict from Full Model

If you have a full model file (not just state dict):

```bash
cd backend
python extract_weights.py
```

This creates `chexca_state_dict.pth` from `chexca_full_model.pth`.

### Customizing UI Colors

Edit `frontend/tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        50: '#eff6ff',
        // ... customize your color palette
      },
    },
    backgroundImage: {
      'gradient-medical': 'linear-gradient(135deg, #0093E9 0%, #80D0C7 50%, #A78BFA 100%)',
    },
  },
}
```

### Adding New Visualizations

1. **Create component** in `frontend/app/components/YourComponent.tsx`
2. **Import in page** at `frontend/app/page.tsx`
3. **Pass result data** as props
4. **Style with Tailwind CSS**

### Modifying Preprocessing

Edit `backend/model_inference.py`:

```python
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Adjust for your training
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Adding New API Endpoints

In `backend/main.py`:

```python
@app.post("/api/your-endpoint")
async def your_endpoint(file: UploadFile = File(...)):
    # Your logic here
    return {"result": "your data"}
```

### Configuring CORS

By default, CORS allows all origins. For production, edit `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-frontend-domain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Security & Privacy

### Data Privacy
- **100% Local Processing** - All analysis happens on your machine
- **No External API Calls** - No data sent to third parties
- **Temporary Storage** - Uploaded images stored in `backend/uploads/` during processing only
- **No Persistent Storage** - Images not saved after analysis (unless you manually save them)
- **HIPAA Consideration** - Suitable for environments requiring local processing

### Security Best Practices

**For Development:**
- Current setup allows all CORS origins (permissive)
- Suitable for local development and testing

**For Production Deployment:**
1. **Restrict CORS** - Limit to specific domains in `backend/main.py`
2. **Add Authentication** - Implement JWT tokens or OAuth
3. **HTTPS Only** - Use SSL/TLS certificates
4. **Rate Limiting** - Prevent abuse with request throttling
5. **Input Validation** - Already implemented (file type checking)
6. **Regular Updates** - Keep dependencies updated for security patches

### File Upload Security

Currently implemented protections:
- File type validation (PNG, JPG, JPEG only)
- File size limits enforced by FastAPI
- Secure filename handling (prevents directory traversal)
- Temporary file cleanup

### Compliance Notes

**Important Medical Disclaimer:**
- This tool is for **research and educational purposes only**
- **NOT FDA approved** for clinical diagnosis
- **NOT a substitute** for professional medical advice
- Results must be **validated by qualified healthcare professionals**
- Users are responsible for ensuring compliance with local healthcare regulations (HIPAA, GDPR, etc.)

## Additional Resources

### Documentation Files
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Quick start guide with common issues
- **[MOCK_MODE.md](MOCK_MODE.md)** - Historical reference (currently disabled)

### Model Training
- Original training used NIH ChestX-ray14 dataset
- Training code implemented: Focal Loss, EMA, Mixed Precision (AMP)
- Architecture: ConvNeXt-Base + CTCA + GAT Fusion
- Contact maintainer for training scripts and notebooks

### Useful Links
- **NIH ChestX-ray14 Dataset:** [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **ConvNeXt Paper:** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- **PyTorch:** [pytorch.org](https://pytorch.org)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Next.js:** [nextjs.org](https://nextjs.org)

### Research Citations

If using this project for research, please cite:

```bibtex
@article{wang2017chestxray14,
  title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2097--2106},
  year={2017}
}
```

## Contributing

This is a research project. If you'd like to contribute:

1. **Report bugs** - Open an issue with detailed reproduction steps
2. **Suggest features** - Describe your use case and proposed solution
3. **Improve documentation** - Submit corrections or clarifications
4. **Share results** - Discuss your findings with the community

## License

This project is provided as-is for educational and research purposes. 

**Important Licensing Notes:**
- Code: Available for educational use
- NIH ChestX-ray14 Dataset: Subject to NIH/Clinical Center terms
- Pre-trained Models: Check respective model licenses (ConvNeXt: MIT)
- No warranty or liability for medical decisions

## Acknowledgments

- **NIH Clinical Center** for the ChestX-ray14 dataset
- **Facebook AI Research** for ConvNeXt architecture
- **PyTorch Team** for the deep learning framework
- **timm library** (Ross Wightman) for model implementations
- **FastAPI** (Sebastián Ramírez) for the backend framework
- **Vercel** for Next.js framework

## Support & Contact

For technical support:
1. Check the **[Troubleshooting](#-troubleshooting)** section
2. Review **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for common issues
3. Consult **[API Documentation](#-api-documentation)** for endpoint details

For research collaborations or advanced questions, contact the project maintainer.

---

<div align="center">

**Built for medical AI research and education**  
**Powered by Deep Learning**  
**Made with care for the healthcare community**

</div>
