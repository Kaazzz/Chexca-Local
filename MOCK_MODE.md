# Mock Mode for Demo

The application is currently in **MOCK MODE** to demonstrate the full 14-class functionality while your model is training.

## Current Status

✅ **MOCK MODE ENABLED**
- Backend returns simulated predictions for all 14 NIH diseases
- Uses sample X-ray with Grad-CAM from `frontend/public/xraygradcam.png`
- Generates realistic co-occurrence matrices
- Perfect for demos and presentations

## What Mock Mode Does

When you upload ANY image:
1. Returns simulated predictions for all 14 diseases
2. Shows realistic probability distributions
3. Displays the sample Grad-CAM heatmap
4. Generates a 14x14 co-occurrence matrix
5. Demonstrates the complete UI

## Switching to Real Model

When your 14-class model is trained and ready:

### Step 1: Update Model Path
Edit `backend/config.py`:
```python
MODEL_PATH = BASE_DIR / "models" / "your_new_14class_model.pth"
```

### Step 2: Disable Mock Mode
Edit `backend/config.py`:
```python
MOCK_MODE = False  # Change from True to False
```

### Step 3: Verify Model Architecture
Make sure your model architecture in `backend/chexca_model.py` matches your trained model.
If needed, update the `CheXCAModel` class to match your architecture.

### Step 4: Restart Backend
```bash
cd backend
python main.py
```

### Step 5: Test
Upload a real X-ray image and verify you get 14 predictions.

## Mock Data Scenarios

The mock mode has predefined scenarios:
- **Scenario 0**: Pneumonia + Infiltration + Consolidation
- **Scenario 1**: Cardiomegaly + Edema + Effusion
- **Scenario 2**: Atelectasis + Infiltration

To change which scenario is used, edit `backend/main.py` line 188:
```python
return get_sample_result(0)  # Change 0 to 1 or 2
```

## Customizing Mock Data

Edit `backend/mock_data.py` to:
- Adjust probability distributions
- Modify co-occurrence patterns
- Change which diseases appear together
- Update sample scenarios

## Current Configuration

- **Mock Mode**: `ENABLED`
- **Sample Image**: `frontend/public/xraygradcam.png`
- **Disease Classes**: 14 NIH ChestX-ray diseases
- **Scenario**: Pneumonia-dominant (default)

---

**Ready for your demo!** The UI will show all 14-class functionality with realistic mock data.
