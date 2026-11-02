# AI-Powered Smart Waste Segregation System

An intelligent automated waste segregation system that combines deep learning, computer vision, and embedded hardware to identify and separate waste into four categories.

The system uses **EfficientNet-B0** for local image classification, Google Gemini API integration, Raspberry Pi 5, Arduino UNO, and automated servo-stepper motor control.

## Project Overview

Improper waste segregation creates environmental pollution and makes recycling difficult. This project aims to automate waste classification and physical segregation using artificial intelligence and embedded systems.

The system identifies waste items and directs them into their designated compartments.

## Waste Categories

| Category | Description |
|---|---|
| PLASTIC | Plastic waste |
| E_WASTE | Electronic waste |
| ORGANIC | Biodegradable waste |
| RECYCLABLE | Recyclable materials |

## Key Features

- AI-based waste classification using EfficientNet-B0.
- Google Gemini API integration.
- Local EfficientNet-B0 classification fallback.
- Raspberry Pi 5 integration.
- Arduino UNO-based motor control.
- Automated four-compartment waste segregation.
- PyQt6-based motion controller interface.
- Limit-switch-based homing and safety mechanisms.
- Dataset cleaning, duplicate detection and evaluation tools.

## System Architecture

The system combines software-based intelligence with hardware-based automation.

**Main processing flow:**

1. Camera captures a waste item.
2. AI classification identifies the waste category.
3. The classification result is mapped to the corresponding bin.
4. Raspberry Pi communicates with the motion controller.
5. Arduino UNO controls the servo and stepper motor.
6. The waste is directed into the selected compartment.

## Technology Stack

### Artificial Intelligence and Software

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- OpenCV
- Google Gemini API
- PyQt6
- NumPy
- Scikit-learn

### Hardware Components

- Raspberry Pi 5
- Arduino UNO
- A4988 Stepper Motor Driver
- NEMA 17 Stepper Motor
- Servo Motor
- USB Camera
- Limit Switches
- Four-compartment waste bin

## Model Performance

The EfficientNet-B0 model was trained and evaluated using a combined dataset of 15,110 images.

### Overall Evaluation Results

| Metric | Result |
|---|---:|
| Validation Accuracy | 97.88% |
| Test Accuracy | 97.44% |
| Correct Test Predictions | 2,208 / 2,266 |
| Macro Precision | 96.84% |
| Macro Recall | 96.79% |
| Macro F1-Score | 96.82% |

### Class-wise F1-Score

| Waste Category | F1-Score |
|---|---:|
| E-WASTE | 99.30% |
| ORGANIC | 98.84% |
| PLASTIC | 91.58% |
| RECYCLABLE | 97.55% |

### Training Configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Image Size | 224 × 224 |
| Epochs | 20 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Loss Function | Weighted Cross-Entropy Loss |

## Dataset Information

The combined dataset contains **15,110 images** collected from multiple publicly available sources.

The dataset was cleaned, organized, deduplicated and divided into training, validation and testing sets.

Raw dataset images are not redistributed in this repository.

For dataset sources and licensing information, refer to:

- `dataset/README.md`
- `dataset/dataset_sources.md`

Please obtain the original datasets from their respective publishers and comply with their licensing terms.

## Repository Structure

```text
AI-Smart-Waste-Segregation-System/
│
├── dataset/
│   ├── README.md
│   └── dataset_sources.md
│
├── hardware/
│   └── arduino/
│       └── waste_motion_controller.ino
│
├── models/
│   └── efficientnet_b0_best.pth
│
├── results/
│   ├── confusion_matrix.png
│   ├── error_analysis.csv
│   ├── evaluation_report.txt
│   ├── evaluation_results.json
│   └── training_history.json
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── camera_predict.py
│   ├── motion_controller_gui.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vmc6433/AI-Smart-Waste-Segregation-System.git

cd AI-Smart-Waste-Segregation-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation may require a platform-specific procedure, particularly on Raspberry Pi 5.

### 5. Configure Gemini API

Create a local `.env` file in the project root directory.

Add:

```env
GEMINI_API_KEY=your_api_key_here
```

Never upload your `.env` file or expose your API key publicly.

### 6. Dataset Setup

The raw datasets are not included in this repository. Obtain the original datasets from their publishers, follow their licensing conditions, and prepare the directory structure expected by the training scripts.

## Hardware and Software Integration

The project integrates:

- Raspberry Pi 5 for AI inference and system-level processing.
- Arduino UNO for motor control and safety logic.
- Servo motor for waste-routing operation.
- NEMA 17 stepper motor for bin positioning.
- Limit switches for homing and movement protection.

## Results and Evaluation

The EfficientNet-B0 model achieved 97.44% test accuracy on the held-out test dataset.

The results demonstrate strong classification performance across the four waste categories. PLASTIC had the lowest class-wise F1-score, indicating scope for further improvement in handling variations in plastic appearance.

## Future Scope

- Expand the dataset with more real-world waste images.
- Improve classification of visually similar waste materials.
- Optimize inference speed on edge devices.
- Integrate additional waste categories.
- Improve mechanical reliability and system scalability.
- Develop remote monitoring and performance analytics.

## Acknowledgements

This project uses publicly available datasets and open-source software libraries. Please refer to the dataset documentation and the respective project licenses for attribution and usage conditions.

## License

License and third-party attribution terms are to be finalized. Until then, please refer to the original dataset and software licenses before reusing project materials.

---

**AI-Powered Smart Waste Segregation System**

Developed as an academic engineering project.
