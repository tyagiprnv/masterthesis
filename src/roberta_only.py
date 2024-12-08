import ast
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter


class RobertaWithMLP(nn.Module):
    def __init__(self, model_name="cardiffnlp/twitter-roberta-large-emotion-latest", num_classes=6, hidden_size=512):
        super(RobertaWithMLP, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        logits = self.mlp(pooled_output)
        return logits


class EmotionDataset(Dataset):
    def __init__(self, data=None, csv_path=None, label_col=None, text_col=None, roberta_tokenizer=None, max_length=512):
        if data is not None:
            self.data = data
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Either `data` or `csv_path` must be provided.")

        self.label_col = label_col
        self.text_col = text_col
        self.tokenizer = roberta_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][self.text_col]
        text_inputs = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        label_tuples = eval(self.data.iloc[idx][self.label_col])  
        label_probs = [prob for _, prob in label_tuples]  
        label_tensor = torch.tensor(label_probs, dtype=torch.float)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
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


def evaluate_model(model, dataloader, device, experiment_dir=None, save_plots=False):
    model.eval()
    total_kl_div = 0.0
    total_cosine_sim = 0.0
    total_mse = 0.0
    num_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask)
            predictions = torch.softmax(predictions, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)

            total_kl_div += F.kl_div(torch.log(predictions), labels, reduction='batchmean').item() * len(labels)
            total_cosine_sim += F.cosine_similarity(predictions, labels, dim=-1).mean().item() * len(labels)
            total_mse += F.mse_loss(predictions, labels, reduction='mean').item() * len(labels)
            num_samples += len(labels)

    avg_kl_div = total_kl_div / num_samples
    avg_cosine_sim = total_cosine_sim / num_samples
    avg_mse = total_mse / num_samples

    if save_plots and experiment_dir:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        plot_and_save_cosine_similarity(predictions_tensor, labels_tensor, f"{experiment_dir}/cosine_similarity.png")
        plot_and_save_label_wise_heatmap(predictions_tensor, labels_tensor, range(predictions_tensor.size(1)), f"{experiment_dir}/label_heatmap.png")

    return avg_kl_div, avg_cosine_sim, avg_mse


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path=None, log_name=None):
    if log_name is None:
        log_name = f"experiment"
    writer = SummaryWriter(log_dir=f"runs/february_exp/{log_name}")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
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

    roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-emotion-latest")
    
    data = pd.read_csv(config["csv_path"])
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=seed)

    train_dataset = EmotionDataset(
        data=train_data,
        label_col=config["label_col"],
        text_col=config["text_col"],
        roberta_tokenizer=roberta_tokenizer
    )

    val_dataset = EmotionDataset(
        data=val_data,
        label_col=config["label_col"],
        text_col=config["text_col"],
        roberta_tokenizer=roberta_tokenizer
    )

    test_dataset = EmotionDataset(
        data=test_data,
        label_col=config["label_col"],
        text_col=config["text_col"],
        roberta_tokenizer=roberta_tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = RobertaWithMLP()

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    experiment_dir = f"models/multimodal_experiments_february/{config['log_name']}"
    os.makedirs(experiment_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/february_exp/{config['log_name']}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")
    
    train_model(
        model, train_loader, val_loader, criterion, optimizer, config["epochs"], 
        device, save_path=f"{experiment_dir}/best_model.pt", log_name=config["log_name"]
    )

    test_kl_div, test_cosine_sim, test_mse = evaluate_model(model, test_loader, device, experiment_dir=experiment_dir, save_plots=True)
    print(f"Test Metrics - KL Div: {test_kl_div:.4f}, Cosine Sim: {test_cosine_sim:.4f}, MSE: {test_mse:.4f}")

    writer.add_scalar("Test/KL_Div", test_kl_div)
    writer.add_scalar("Test/Cosine_Sim", test_cosine_sim)
    writer.add_scalar("Test/MSE", test_mse)

    writer.close()


def main():
    epochs_list = [2, 5]
    log_name_base = "exp_only_roberta_large_lr1e-5"

    common_params = {
        "learning_rate": 1e-5,
        "csv_path": "/work/ptyagi/masterthesis/data/predictions/feb/averaged_predictions.csv",
        "image_dir": "/work/ptyagi/ClimateVisions/Images/2019/02_February",
        "label_col": "averaged_predictions",
        "text_col": "tweet_text",
        "image_col": "matched_filename"
    }

    configs = []
    for epochs in epochs_list:
        log_name = f"{log_name_base}_epochs{epochs}"

        config = {
            "epochs": epochs,
            "log_name": log_name,
            **common_params
            }
        configs.append(config)

    seed = 42 

    for config in configs:
        train_and_evaluate(config, seed=seed)


if __name__ == "__main__":
    main()