import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image


# ============================================================
# PART 2 - FAST PRODUCT IMAGE CATEGORISER
# Fashion-MNIST + ResNet18 Transfer Learning
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
MODEL_DIR = "models"
SAMPLE_DIR = "data/sample_images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

CLASS_NAMES = [
    "tshirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]

NUM_CLASSES = 10
BATCH_SIZE = 256

# Faster than 224x224 while still suitable for transfer learning
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading Fashion-MNIST dataset...")

train_full = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_full = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

print("Full training images:", len(train_full))
print("Full test images:", len(test_full))


# ============================================================
# 2. FAST TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

train_indices = rng.choice(
    len(train_full),
    size=10000,
    replace=False
)

remaining = np.setdiff1d(
    np.arange(len(train_full)),
    train_indices
)

val_indices = rng.choice(
    remaining,
    size=2000,
    replace=False
)

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    train_full,
    val_indices
)

# Use 2,000 untouched test images for fast evaluation
test_indices = np.arange(2000)

test_dataset = Subset(
    test_full,
    test_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training split:", len(train_dataset))
print("Validation split:", len(val_dataset))
print("Test split:", len(test_dataset))


# ============================================================
# 3. PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)

# Freeze pretrained backbone
for parameter in resnet.parameters():
    parameter.requires_grad = False

feature_extractor = copy.deepcopy(resnet)

# Remove original ImageNet classifier
feature_extractor.fc = nn.Identity()

feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

print("Device:", DEVICE)
print("Model: ResNet18")
print("Backbone frozen: YES")
print("Transfer learning: YES")


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def extract_features(loader, name):

    features = []
    labels = []

    total = len(loader)

    with torch.no_grad():

        for batch_no, (images, batch_labels) in enumerate(
            loader, start=1
        ):

            images = images.to(DEVICE)

            batch_features = feature_extractor(images)

            features.append(
                batch_features.cpu()
            )

            labels.append(
                batch_labels
            )

            if batch_no % 5 == 0 or batch_no == total:
                print(
                    f"{name}: "
                    f"{batch_no}/{total} batches"
                )

    return (
        torch.cat(features),
        torch.cat(labels)
    )


print("\nExtracting cached features...")

train_features, train_labels = extract_features(
    train_loader,
    "TRAIN"
)

val_features, val_labels = extract_features(
    val_loader,
    "VALIDATION"
)

test_features, test_labels = extract_features(
    test_loader,
    "TEST"
)

print(
    "Feature shape:",
    train_features.shape
)


# ============================================================
# 5. TRAIN SMALL CLASSIFIER
# ============================================================

classifier = nn.Linear(
    train_features.shape[1],
    NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)

train_features = train_features.to(DEVICE)
train_labels = train_labels.to(DEVICE)

val_features = val_features.to(DEVICE)
val_labels = val_labels.to(DEVICE)

best_accuracy = 0.0
best_state = copy.deepcopy(
    classifier.state_dict()
)

print("\nTraining classifier...")

for epoch in range(10):

    classifier.train()

    optimizer.zero_grad()

    outputs = classifier(
        train_features
    )

    loss = criterion(
        outputs,
        train_labels
    )

    loss.backward()
    optimizer.step()

    classifier.eval()

    with torch.no_grad():

        val_outputs = classifier(
            val_features
        )

        val_predictions = val_outputs.argmax(
            dim=1
        )

        val_accuracy = (
            val_predictions == val_labels
        ).float().mean().item()

    print(
        f"Epoch {epoch + 1}/10 "
        f"- Loss: {loss.item():.4f} "
        f"- Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_state = copy.deepcopy(
            classifier.state_dict()
        )


classifier.load_state_dict(
    best_state
)

print(
    "\nBest validation accuracy:",
    round(best_accuracy, 4)
)


# ============================================================
# 6. TEST EVALUATION
# ============================================================

classifier.eval()

test_features = test_features.to(DEVICE)
test_labels = test_labels.to(DEVICE)

with torch.no_grad():

    test_outputs = classifier(
        test_features
    )

    test_predictions = test_outputs.argmax(
        dim=1
    )

test_accuracy = accuracy_score(
    test_labels.cpu().numpy(),
    test_predictions.cpu().numpy()
)

print(
    "\nFINAL TEST ACCURACY:",
    round(test_accuracy, 4)
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

all_labels = test_labels.cpu().numpy()
all_predictions = test_predictions.cpu().numpy()

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX:")
print(cm)

cm_copy = cm.copy()

np.fill_diagonal(
    cm_copy,
    0
)

pairs = []

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        if cm_copy[i, j] > 0:

            pairs.append(
                (
                    cm_copy[i, j],
                    CLASS_NAMES[i],
                    CLASS_NAMES[j]
                )
            )

pairs.sort(reverse=True)

print("\nTop confusion pairs:")

for count, actual, predicted in pairs[:5]:

    print(
        f"{actual} -> {predicted}: "
        f"{count} images"
    )


# ============================================================
# 9. SAVE MODEL ARTIFACT
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "product_classifier.pt"
)

torch.save(
    {
        "feature_extractor_state_dict":
            feature_extractor.state_dict(),

        "classifier_state_dict":
            classifier.state_dict(),

        "class_names":
            CLASS_NAMES,

        "architecture":
            "resnet18_transfer_learning",

        "image_size":
            64,

        "num_classes":
            NUM_CLASSES
    },
    model_path
)

print("\nMODEL SAVED SUCCESSFULLY")
print(
    "Saved to:",
    model_path
)


# ============================================================
# 10. SAVE 5+ REAL PNG IMAGES
# ============================================================

print("\nSaving real Fashion-MNIST PNG images...")

raw_test = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=False
)

saved_classes = set()

for index in range(len(raw_test)):

    image, label = raw_test[index]

    if label not in saved_classes:

        filename = (
            f"{label:02d}_"
            f"{CLASS_NAMES[label]}.png"
        )

        filepath = os.path.join(
            SAMPLE_DIR,
            filename
        )

        image.save(filepath)

        saved_classes.add(label)

        print(
            "Saved:",
            filepath
        )

    if len(saved_classes) >= 5:
        break

print(
    "\nSample PNG count:",
    len(saved_classes)
)


# ============================================================
# 11. IMAGE CLASSIFICATION FUNCTION
# ============================================================

def classify_product_image(image_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    inference_extractor = models.resnet18(
        weights=None
    )

    inference_extractor.fc = nn.Identity()

    inference_extractor.load_state_dict(
        checkpoint[
            "feature_extractor_state_dict"
        ]
    )

    inference_extractor = (
        inference_extractor.to(DEVICE)
    )

    inference_extractor.eval()

    inference_classifier = nn.Linear(
        512,
        checkpoint["num_classes"]
    ).to(DEVICE)

    inference_classifier.load_state_dict(
        checkpoint[
            "classifier_state_dict"
        ]
    )

    inference_classifier.eval()

    image = Image.open(
        image_path
    ).convert("L")

    image = transform(image)

    image = image.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        features = inference_extractor(
            image
        )

        output = inference_classifier(
            features
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

    return {
        "label":
            checkpoint[
                "class_names"
            ][predicted_class],

        "confidence":
            float(
                probabilities[
                    0,
                    predicted_class
                ]
            )
    }


print("\nPart 2 completed successfully!")