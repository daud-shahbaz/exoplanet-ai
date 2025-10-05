# 🌌 NASA Exoplanet Detection System

AI-powered system to detect exoplanets using NASA's Kepler telescope data. Built for **NASA Space Apps Challenge 2025**.

**Live Demo**: [https://www.youtube.com/watch?v=-9CzAftSwL0](https://github.com/daud-shahbaz/exoplanet-ai)

---

## 👥 Team

1. **Daud Shahbaz** - Team Lead
2. **Abdul Muizz** - Strategy Lead
3. **Aswad Sheeraz** - AI Lead
4. **Sameed Irfan** - Web Lead
5. **Abiha Azhar** - Documentation Lead
6. **Maheen Sajid** - Demo Lead

---

## 🎯 What It Does

Detects exoplanets by analyzing 10 key factors from telescope data:
- Orbital period, transit depth/duration
- Planet radius and temperature
- Star properties (temperature, gravity, radius)
- Impact parameter

**Accuracy**: 82.49% using ensemble of 3 deep learning models (DNN + CNN + LSTM)

---

## 🚀 Quick Start

### Install
```bash
# Clone repository
git clone https://github.com/daud-shahbaz/exoplanet-ai.git
cd exoplanet-ai

# Create virtual environment
python -m venv nasa
nasa\Scripts\activate  # Windows
source nasa/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Data
Get NASA datasets from [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and place in `dataset/` folder:
- `cumulative_2025.10.05_02.33.52.csv`
- `TOI_2025.10.05_02.34.10.csv`
- `k2pandc_2025.10.05_03.01.55.csv`

### Train Models
```bash
python train_models.py 
```

### Run Application
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## 💻 Usage

### Manual Input Tab
- Adjust 10 sliders for planet parameters
- Click "Detect Exoplanet"
- View prediction, confidence, and model scores

### Upload File Tab
- Upload CSV with planet data
- Get batch analysis results

### Visualizations Tab
- View feature importance charts
- Compare model performance

---

## 📊 Performance

| Model | Accuracy |
|-------|----------|
| DNN   | 80.45%   |
| CNN   | 82.54%   |
| LSTM  | 79.19%   |
| **Ensemble** | **82.49%** |

**ROC-AUC Score**: 0.90

---

## 📁 Project Structure

```
├── app.py                  # Flask web application
├── train_models.py         # Model training script
├── requirements.txt        # Dependencies
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── css/style.css      # Styling
│   └── js/app.js          # Frontend logic
├── models/                # Trained models (DNN, CNN, LSTM)
└── dataset/               # NASA data files
```

---

## 🛠️ Tech Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript, Plotly
- **Data**: Pandas, NumPy, Scikit-learn
- **Models**: Deep Neural Network, CNN, LSTM

---

## 📧 Contact

**Team Lead**: Daud Shahbaz  
**GitHub**: [github.com/daud-shahbaz/exoplanet-ai](https://github.com/daud-shahbaz/exoplanet-ai)

---

**NASA Space Apps Challenge 2025**
