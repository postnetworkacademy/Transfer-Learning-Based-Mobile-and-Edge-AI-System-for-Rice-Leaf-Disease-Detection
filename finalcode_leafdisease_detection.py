# Research Paper Link https://doi.org/10.5281/zenodo.17909033
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
from tqdm import tqdm

# ---------------------- CONFIG ----------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 6
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------- PRETTY NAMES ----------------------
pretty_names = {
    "mobilenet_v2": "MobileNet",
    "shufflenet_v2_x1_0": "ShuffleNet",
    "squeezenet1_0": "SqueezeNet"
}

# ---------------------- DATA PREPROCESSING ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("rice_leaf_dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("rice_leaf_dataset/validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------- MODEL DEFINITIONS ----------------------
def get_model(name):
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
        model.num_classes = NUM_CLASSES
    else:
        raise ValueError("Unknown model name")
    return model.to(device)

# ---------------------- CHECKPOINT UTILS ----------------------
def checkpoint_dir():
    os.makedirs("checkpoints", exist_ok=True)
    return "checkpoints"

def get_checkpoint_paths(model_name):
    base = os.path.join(checkpoint_dir(), model_name)
    return {
        "history": base + "_history.json",
        "weights": base + "_weights.pt"
    }

def save_checkpoint(history, model, model_name):
    paths = get_checkpoint_paths(model_name)
    with open(paths["history"], "w") as f:
        json.dump(history, f, indent=4)
    torch.save(model.state_dict(), paths["weights"])
    print(f"💾 Checkpoint saved: {paths['history']}, {paths['weights']}")

def load_checkpoint_if_exists(model, model_name):
    paths = get_checkpoint_paths(model_name)
    if os.path.exists(paths["history"]):
        with open(paths["history"], "r") as f:
            history = json.load(f)
        print(f"✅ History loaded from checkpoint: {paths['history']}")
    else:
        history = {
            "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
            "precision": [], "recall": [], "f1": [], "roc_auc": []
        }

    if os.path.exists(paths["weights"]):
        model.load_state_dict(torch.load(paths["weights"], map_location=device))
        print(f"✅ Model weights loaded from checkpoint: {paths['weights']}")
    return history

# ---------------------- TRAIN & EVALUATE ----------------------
def train_and_evaluate(model, train_loader, test_loader, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = load_checkpoint_if_exists(model, model_name)
    start_epoch = len(history["train_loss"])

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{pretty_names[model_name]}]"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss, val_corrects = 0.0, 0
            all_preds, all_labels, all_probs = [], [], []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            val_epoch_loss = val_loss / len(test_loader.dataset)
            val_epoch_acc = accuracy_score(all_labels, all_preds)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="weighted"
            )

            try:
                roc_auc = roc_auc_score(
                    np.eye(NUM_CLASSES)[all_labels], np.array(all_probs), multi_class="ovr"
                )
            except:
                roc_auc = float("nan")

            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc.item())
            history["test_loss"].append(val_epoch_loss)
            history["test_acc"].append(val_epoch_acc)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1"].append(f1)
            history["roc_auc"].append(roc_auc)

            print(f"[{pretty_names[model_name]}] Epoch {epoch+1}/{EPOCHS} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, "
                  f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

            # Save checkpoint after every 2 epochs
            if (epoch + 1) % 2 == 0:
                save_checkpoint(history, model, model_name)

            # Optional: Save on last epoch too
            if epoch == EPOCHS - 1:
                save_checkpoint(history, model, model_name)

    except KeyboardInterrupt:
        print(f"⛔ Training interrupted at epoch {epoch+1}. Saving checkpoint.")
        save_checkpoint(history, model, model_name)

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}. Saving checkpoint.")
        save_checkpoint(history, model, model_name)

    return history
# ---------------------- PRETTY METRIC NAMES ----------------------
pretty_metrics = {
    "train_loss": "Training Loss",
    "test_loss": "Validation Loss",
    "train_acc": "Training Accuracy",
    "test_acc": "Validation Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC-AUC"
}

# ---------------------- PLOTTING ----------------------
def plot_comparison(all_stats, model_names, metric):
    plt.figure(figsize=(10, 7))
    for model in model_names:
        stats = all_stats[model]
        plt.plot(range(1, len(stats[metric]) + 1), stats[metric], marker='o', label=pretty_names[model],linewidth=4.5)
    #plt.title(f'{metric.capitalize()} Comparison Across Models')
    plt.title(f'{pretty_metrics[metric]} Comparison Across Models')
    plt.xlabel('Epoch')
    #plt.ylabel(metric.capitalize())
    plt.ylabel(pretty_metrics[metric])
    plt.grid(True)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    filename = f'plots/{metric}_comparison.png'
    plt.savefig(filename)
    print(f"📊 Plot saved: {filename}")
    plt.show()
    plt.close()

# ---------------------- MAIN SCRIPT ----------------------
if __name__ == "__main__":
    model_names = ["mobilenet_v2", "shufflenet_v2_x1_0", "squeezenet1_0"]
    all_stats = {}

    for name in model_names:
        model = get_model(name)
        stats = train_and_evaluate(model, train_loader, test_loader, name)
        all_stats[name] = stats

    for metric in ["train_loss", "train_acc", "test_loss", "test_acc", "precision", "recall", "f1", "roc_auc"]:
        plot_comparison(all_stats, model_names, metric)

