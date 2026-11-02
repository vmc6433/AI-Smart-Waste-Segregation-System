from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "efficientnet_b0_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print("=" * 70)
    print("AI WASTE SEGREGATION SYSTEM")
    print("IMAGE PREDICTION")
    print("=" * 70)

    print("\nUsage:")
    print(
        "python scripts/predict.py "
        "\"path_to_image.jpg\""
    )

    print("\nExample:")
    print(
        "python scripts/predict.py "
        "\"C:\\Users\\vchet\\Desktop\\plastic.jpg\""
    )

    sys.exit(1)


IMAGE_PATH = Path(sys.argv[1])


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# MODEL WEIGHTS / TRANSFORM
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("AI WASTE SEGREGATION SYSTEM")
print("IMAGE PREDICTION")
print("=" * 70)

print(f"Image : {IMAGE_PATH}")
print(f"Model : {MODEL_PATH}")
print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU   : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("-" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

num_classes = len(class_names)

model = efficientnet_b0(
    weights=None
)

in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

except Exception as e:

    raise RuntimeError(
        f"Could not open image: {e}"
    )


input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        input_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = (
        class_names[predicted_index]
    )

    confidence = (
        probabilities[predicted_index]
        .item()
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"Predicted Class : "
    f"{predicted_class}"
)

print(
    f"Confidence      : "
    f"{confidence * 100:.2f}%"
)

print("\nCLASS PROBABILITIES")
print("-" * 70)

sorted_predictions = sorted(
    zip(
        class_names,
        probabilities.cpu().numpy()
    ),
    key=lambda x: x[1],
    reverse=True
)

for class_name, probability in sorted_predictions:

    print(
        f"{class_name:15s} : "
        f"{probability * 100:6.2f}%"
    )


print("=" * 70)