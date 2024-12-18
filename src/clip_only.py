import ast
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter


class ClipWithMLP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", num_classes=6, hidden_size=512):
        super(ClipWithMLP, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_features = self.clip_model.get_image_features(image)
        logits = self.mlp(image_features)
        return logits


class EmotionDataset(Dataset):
    def __init__(self, data=None, csv_path=None, label_col=None, image_col=None, image_dir=None, clip_processor=None):
        if data is not None:
            self.data = data
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Either `data` or `csv_path` must be provided.")

        self.label_col = label_col
        self.image_col = image_col
        self.image_dir = image_dir
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx][self.image_col]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        label_tuples = eval(self.data.iloc[idx][self.label_col])  
        label_probs = [prob for _, prob in label_tuples]  
        label_tensor = torch.tensor(label_probs, dtype=torch.float)

        return {
            "image": image,
            "labels": label_tensor,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_and_save_cosine_similarity(predictions, labels, save_path):
    similarities = F.cosine_similarity(predictions, labels, dim=-1).detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.kdeplot(similarities, fill=True, color="blue")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.axvline(x=1.0, color='green', linestyle='--', label='Perfect Similarity')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_and_save_label_wise_heatmap(predictions, labels, label_order, save_path):
    avg_pred = predictions.mean(dim=0).detach().cpu().numpy()
    avg_label = labels.mean(dim=0).cpu().numpy()
    heatmap_data = np.vstack([avg_label, avg_pred])
    row_labels = ["True", "Predicted"]
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=label_order, yticklabels=row_labels)
    plt.title("Label-Wise Average Probabilities")
    plt.xlabel("Labels")
    plt.ylabel("")
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, dataloader, device, experiment_dir=None, save_plots=False, writer=None):
    model.eval()
    total_kl_div = 0.0
    total_cosine_sim = 0.0
    total_mse = 0.0
    num_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(images)
            predictions = torch.softmax(predictions, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)

            total_kl_div += F.kl_div(torch.log(predictions), labels, reduction='batchmean').item() * len(images)
            total_cosine_sim += F.cosine_similarity(predictions, labels, dim=-1).mean().item() * len(images)
            total_mse += F.mse_loss(predictions, labels, reduction='mean').item() * len(images)
            num_samples += len(images)

    avg_kl_div = total_kl_div / num_samples
    avg_cosine_sim = total_cosine_sim / num_samples
    avg_mse = total_mse / num_samples

    if save_plots and experiment_dir:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        pred_labels = predictions_tensor.argmax(dim=-1).cpu().numpy()
        true_labels = labels_tensor.argmax(dim=-1).cpu().numpy()

        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)

        plot_and_save_cosine_similarity(predictions_tensor, labels_tensor, f"{experiment_dir}/cosine_similarity.png")
        plot_and_save_label_wise_heatmap(predictions_tensor, labels_tensor, range(predictions_tensor.size(1)), f"{experiment_dir}/label_heatmap.png")
    
    if writer:
        writer.add_scalar("Test/KL_Div", avg_kl_div)
        writer.add_scalar("Test/Cosine_Sim", avg_cosine_sim)
        writer.add_scalar("Test/MSE", avg_mse)
        writer.add_scalar("Test/Accuracy", accuracy)
        writer.add_scalar("Test/Precision", precision)
        writer.add_scalar("Test/Recall", recall)

    return avg_kl_div, avg_cosine_sim, avg_mse


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path=None, log_name=None):
    if log_name is None:
        log_name = f"experiment"
    writer = SummaryWriter(log_dir=f"runs/august_exp/{log_name}")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")

        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(torch.log_softmax(outputs, dim=-1), labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        val_kl_div, val_cosine_sim, val_mse = evaluate_model(model, val_loader, device, save_plots=False)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Metrics - KL Div: {val_kl_div:.4f}, Cosine Sim: {val_cosine_sim:.4f}, MSE: {val_mse:.4f}")

        writer.add_scalar("Loss/Val_KL_Div", val_kl_div, epoch)
        writer.add_scalar("Metrics/Val_Cosine_Sim", val_cosine_sim, epoch)
        writer.add_scalar("Metrics/Val_MSE", val_mse, epoch)

        if val_kl_div < best_val_loss:
            best_val_loss = val_kl_div
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at {save_path}")

    writer.close()
    print(f"Logs saved to runs/{log_name}")


def train_and_evaluate(config, seed=42):
    set_seed(seed)
    print(config['log_name'])

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    data = pd.read_csv(config["csv_path"])
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=seed)

    train_dataset = EmotionDataset(
        data=train_data,
        label_col=config["label_col"],
        image_col=config["image_col"],
        image_dir=config["image_dir"],
        clip_processor=clip_processor
    )

    val_dataset = EmotionDataset(
        data=val_data,
        label_col=config["label_col"],
        image_col=config["image_col"],
        image_dir=config["image_dir"],
        clip_processor=clip_processor
    )

    test_dataset = EmotionDataset(
        data=test_data,
        label_col=config["label_col"],
        image_col=config["image_col"],
        image_dir=config["image_dir"],
        clip_processor=clip_processor
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ClipWithMLP()
    
    if config.get("freeze_clip", False):
        model.freeze_clip()

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    experiment_dir = f"models/multimodal_experiments_august/{config['log_name']}"
    os.makedirs(experiment_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/august_exp/{config['log_name']}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")
    
    train_model(
        model, train_loader, val_loader, criterion, optimizer, config["epochs"], 
        device, save_path=f"{experiment_dir}/best_model.pt", log_name=config["log_name"]
    )

    test_kl_div, test_cosine_sim, test_mse = evaluate_model(model, test_loader, device, experiment_dir=experiment_dir, save_plots=True, writer=writer)
    print(f"Test Metrics - KL Div: {test_kl_div:.4f}, Cosine Sim: {test_cosine_sim:.4f}, MSE: {test_mse:.4f}")

    writer.close()


def main():
    epochs_list = [10]
    log_name_base = "exp_only_clip_lr1e-5"

    common_params = {
        "learning_rate": 1e-5,
        "csv_path": "/work/ptyagi/masterthesis/data/predictions/aug/averaged_predictions.csv",
        "image_dir": "/work/ptyagi/ClimateVisions/Images/2019/08_August",
        "label_col": "averaged_predictions",
        "text_col": "tweet_text",
        "image_col": "matched_filename"
    }

    configs = []
    seed = 42
    for epochs in epochs_list:
        log_name = f"{log_name_base}_epochs{epochs}_seed{seed}_frozen"

        config = {
            "epochs": epochs,
            "log_name": log_name,
            "freeze_clip" : True,
            **common_params
            }
        configs.append(config) 

    for config in configs:
        train_and_evaluate(config, seed=seed)


if __name__ == "__main__":
    main()