
# AI-Powered Smart Waste Segregation System

An AI-powered automated waste segregation system that classifies waste into four categories using EfficientNet-B0 and Google Gemini, then automatically routes the waste using a Raspberry Pi, Arduino UNO, servo motor and NEMA 17 stepper motor.

## Waste Categories

- Plastic
- E-Waste
- Organic
- Recyclable

## Key Features

- Deep learning-based waste classification using EfficientNet-B0.
- Google Gemini API integration.
- Local AI fallback for classification.
- Raspberry Pi 5 integration.
- Arduino UNO-based motor control.
- Automated four-bin waste segregation.
- PyQt6 motion controller interface.
- Limit-switch-based homing and safety controls.

## Technology Stack

### AI and Software
- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- OpenCV
- Google Gemini API
- PyQt6

### Hardware
- Raspberry Pi 5
- Arduino UNO
- A4988 Stepper Driver
- NEMA 17 Stepper Motor
- Servo Motor
- Limit Switches

## Model Performance

| Metric | Result |
|---|---:|
| Validation Accuracy | 97.88% |
| Test Accuracy | 97.44% |
| Macro Precision | 96.84% |
| Macro Recall | 96.79% |
| Macro F1-score | 96.82% |

The model was trained for 20 epochs using EfficientNet-B0.

## Dataset

The project uses a combined dataset containing 15,110 images from multiple publicly available sources.

The dataset was organized into four categories:
- PLASTIC
- E_WASTE
- ORGANIC
- RECYCLABLE

Raw dataset images are not redistributed in this repository. Refer to the dataset documentation for original sources and licensing information.

## Repository Structure

- `models/` - Trained model checkpoint.
- `scripts/` - Python training, evaluation and application code.
- `hardware/` - Arduino firmware and hardware resources.
- `results/` - Model evaluation results.
- `dataset/` - Dataset documentation.

## Installation

Clone the repository:

```bash
git clone https://github.com/vmc6433/AI-Smart-Waste-Segregation-System.git
cd AI-Smart-Waste-Segregation-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key in a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit your `.env` file.

## License

License and third-party dataset attribution details will be documented separately.
