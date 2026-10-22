# PawCare AI — Dog Skin Disease Classifier 🐾

A premium, production-style **Streamlit** web application that uses a fine-tuned **MobileNetV2** deep-learning model to classify canine skin conditions from uploaded photos.

---

## Features

- **AI-Powered Analysis** — MobileNetV2 classifier trained on veterinary dermatology images.
- **Four Condition Categories** — Bacterial Dermatosis, Fungal Infections, Healthy, and Hypersensitivity / Allergic Dermatosis.
- **Modern Dashboard UI** — Forest-green, ivory, and sage color palette with professional veterinary aesthetic.
- **Instant Results** — Upload → Analyze → Review class-by-class score breakdowns.
- **Responsible AI** — Clearly labeled as experimental with visible disclaimers throughout.

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested with 3.12)
- macOS (Apple Silicon compatible), Linux, or Windows

### Installation

```bash
# Clone / navigate to the project
cd dog-skin-ai

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## Project Structure

```
dog-skin-ai/
├── app.py                      # Streamlit application
├── best_mobilenetv2.keras      # Pre-trained MobileNetV2 model (11 MB)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## Model Details

| Property       | Value                                    |
|----------------|------------------------------------------|
| Architecture   | MobileNetV2 (transfer learning)          |
| Input          | 224 × 224 × 3 (RGB)                     |
| Output         | 4 classes (softmax)                      |
| Format         | Keras v3 (`.keras`)                      |
| Size           | ~11 MB                                   |

### Class Labels (fixed order)

1. **Bacterial Dermatosis** — Bacterial skin infections
2. **Fungal Infections** — Fungal conditions (e.g., ringworm)
3. **Healthy** — No significant skin disease detected
4. **Hypersensitivity / Allergic Dermatosis** — Allergic reactions

---

## ⚠️ Disclaimer

> **Experimental AI result only. This tool cannot diagnose skin conditions. Consult a qualified veterinarian.**
>
> This application is a research and educational tool. It is **not** a medical device, has **not** been clinically validated, and should **never** replace professional veterinary consultation. Do not delay seeking professional treatment based on this tool's output.

---

## License

This project is provided for educational and research purposes.

