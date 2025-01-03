import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from torchmultimodal.models.flava.model import flava_model_for_classification
from transformers import AutoProcessor


from sklearn.model_selection import train_test_split


class FLAVAMultiModalDataset(Dataset):
    """
    Adapted for FLAVA, using FLAVAImageTransform and FLAVATextTransform.
    Expects 'data' containing columns with image filenames, text, and labels.
    """

    def __init__(
        self,
        data=None,
        csv_path=None,
        image_dir=None,
        label_col=None,
        image_col=None,
        text_col=None,
        max_text_length=128,
    ):
        if data is not None:
            self.data = data
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Either 'data' or 'csv_path' must be provided.")

        self.image_dir = image_dir
        self.label_col = label_col
        self.image_col = image_col
        self.text_col = text_col

        # FLAVA transforms:
        self.image_transform = FLAVAImageTransform(
            # FLAVA typically uses 224x224, you can change if needed
            image_size=224  
        )
        self.text_transform = FLAVATextTransform(
            max_seq_len=max_text_length,
            # You can adjust these, but for classification, 
            # we typically do not want random masking
            mask_probability=0.0,  
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1) Load and transform image
        image_filename = row[self.image_col]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)  # returns a tensor

        # 2) Transform text
        text = row[self.text_col]
        text_inputs = self.text_transform(text)
        # text_inputs is a dict with keys like: 'input_ids', 'attention_mask'
        # Each is already a tensor of shape [max_seq_len]

        # 3) Load label
        # Here, the label is a list of (label_name, probability) pairs
        # e.g. [('joy', 0.8), ('sadness', 0.2)] => we only want probabilities
        # or something similar with 6 labels
        label_tuples = eval(row[self.label_col])  # e.g. "[('joy', 0.2), ...]"
        label_probs = [prob for _, prob in label_tuples]
        label_tensor = torch.tensor(label_probs, dtype=torch.float)

        return {
            "image": image_tensor,  # shape [3, 224, 224]
            "input_ids": text_inputs["input_ids"],         # shape [max_seq_len]
            "attention_mask": text_inputs["attention_mask"],# shape [max_seq_len]
            "labels": label_tensor,                         # shape [num_labels]
        }


# ---------------------
# Model Definition
# ---------------------
class FLAVAForEmotion(nn.Module):
    """
    A simple model that uses FLAVA to produce a multimodal embedding,
    then projects it to num_labels for emotion prediction.
    """
    def __init__(self, num_labels=6, hidden_size=768):
        super().__init__()
        self.config = FlavaConfig()

        # Initialize FLAVA
        # If you have a pretrained checkpoint, 
        # you could do FlavaModel.from_pretrained("facebook/flava-full")
        # For brevity, we'll instantiate from config
        self.flava = FlavaModel(self.config)

        # The default hidden dimension for FLAVA base is typically 768
        # Adjust as needed if you have a different config
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, image, input_ids, attention_mask):
        """
        image: shape [batch_size, 3, 224, 224]
        input_ids: shape [batch_size, seq_len]
        attention_mask: shape [batch_size, seq_len]
        """
        outputs = self.flava(
            # FLAVA expects these arguments:
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # You get multiple outputs: text_embeddings, image_embeddings, multimodal_embeddings, etc.
        # Typically, we use the final multimodal output for classification.
        # FLAVA returns a dictionary. The key for the final multimodal output is usually 'multimodal_output',
        # shape [batch_size, hidden_size].
        multimodal_embeds = outputs.multimodal_output

        # Pass to a classification head
        logits = self.classifier(multimodal_embeds)
        return logits

    def freeze_flava_vision(self):
        """
        Example method to freeze the vision trunk if desired.
        """
        for param in self.flava.visual_encoder.parameters():
            param.requires_grad = False

    def freeze_flava_text(self):
        """
        Example method to freeze the text trunk if desired.
        """
        for param in self.flava.text_encoder.parameters():
            param.requires_grad = False


# ---------------------
# Utility functions
# ---------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_initial_weights(model):
    # For demonstration, only saving the entire model’s parameters.
    # If you want separate parts, adapt accordingly.
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    return initial_weights


def compare_weights(initial_weights, model):
    weight_differences = {}
    for name, param in model.named_parameters():
        weight_differences[name] = (param - initial_weights[name]).norm().item()
    return weight_differences


def plot_weight_updates(weight_differences, model_name, save_path):
    updates = list(weight_differences.values())
    plt.figure(figsize=(8, 6))
    plt.hist(updates, bins=50, alpha=0.7, color="blue")
    plt.title(f"Weight Updates for {model_name}")
    plt.xlabel("Weight Update Magnitude (L2 Norm)")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


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
    sns.heatmap(
        heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
        xticklabels=label_order, yticklabels=row_labels
    )
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
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images, input_ids, attention_mask)
            predictions = torch.softmax(logits, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)

            # Accumulate metrics
            # KL Div
            total_kl_div += F.kl_div(torch.log(predictions), labels, reduction='batchmean').item() * len(images)
            # Cosine similarity (average over batch)
            total_cosine_sim += F.cosine_similarity(predictions, labels, dim=-1).mean().item() * len(images)
            # MSE
            total_mse += F.mse_loss(predictions, labels, reduction='mean').item() * len(images)

            num_samples += len(images)

    avg_kl_div = total_kl_div / num_samples
    avg_cosine_sim = total_cosine_sim / num_samples
    avg_mse = total_mse / num_samples

    if save_plots and experiment_dir:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        plot_and_save_cosine_similarity(
            predictions_tensor, labels_tensor,
            save_path=f"{experiment_dir}/cosine_similarity.png"
        )
        plot_and_save_label_wise_heatmap(
            predictions_tensor, labels_tensor,
            label_order=range(predictions_tensor.size(1)),
            save_path=f"{experiment_dir}/label_heatmap.png"
        )

    return avg_kl_div, avg_cosine_sim, avg_mse


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device,
                save_path=None, log_name=None):
    if log_name is None:
        log_name = "experiment"
    writer = SummaryWriter(log_dir=f"runs/flava_exp/{log_name}")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")

        for batch in progress_bar:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        val_kl_div, val_cosine_sim, val_mse = evaluate_model(model, val_loader, device, save_plots=False)
        print(
            f"Epoch {epoch + 1}/{epochs} - Validation Metrics "
            f"- KL Div: {val_kl_div:.4f}, Cosine Sim: {val_cosine_sim:.4f}, MSE: {val_mse:.4f}"
        )
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

    # Read data
    data = pd.read_csv(config["csv_path"])
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=seed)

    # Prepare datasets
    train_dataset = FLAVAMultiModalDataset(
        data=train_data,
        image_dir=config["image_dir"],
        label_col=config["label_col"],
        image_col=config["image_col"],
        text_col=config["text_col"],
        max_text_length=128,
    )
    val_dataset = FLAVAMultiModalDataset(
        data=val_data,
        image_dir=config["image_dir"],
        label_col=config["label_col"],
        image_col=config["image_col"],
        text_col=config["text_col"],
        max_text_length=128,
    )
    test_dataset = FLAVAMultiModalDataset(
        data=test_data,
        image_dir=config["image_dir"],
        label_col=config["label_col"],
        image_col=config["image_col"],
        text_col=config["text_col"],
        max_text_length=128,
    )

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize FLAVA-based model
    model = FLAVAForEmotion(
        num_labels=config["num_labels"],
        hidden_size=config["hidden_size"],
    )

    # Optional freeze if you'd like
    if config.get("freeze_flava_vision", False):
        model.freeze_flava_vision()
    if config.get("freeze_flava_text", False):
        model.freeze_flava_text()

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Where to store logs/models
    experiment_dir = f"models/flava_experiments/{config['log_name']}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save initial weights (for analysis)
    initial_weights = save_initial_weights(model)

    writer = SummaryWriter(log_dir=f"runs/flava_exp/{config['log_name']}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")

    # Train
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config["epochs"],
        device,
        save_path=f"{experiment_dir}/best_model.pt",
        log_name=config["log_name"]
    )

    # Compare weights
    weight_differences = compare_weights(initial_weights, model)
    plot_weight_updates(weight_differences, "FLAVA", save_path=f"{experiment_dir}/flava_weight_updates.png")

    # Test
    test_kl_div, test_cosine_sim, test_mse = evaluate_model(
        model,
        test_loader,
        device,
        experiment_dir=experiment_dir,
        save_plots=True,
    )
    print(
        f"Test Metrics - KL Div: {test_kl_div:.4f}, "
        f"Cosine Sim: {test_cosine_sim:.4f}, MSE: {test_mse:.4f}"
    )

    writer.add_scalar("Test/KL_Div", test_kl_div)
    writer.add_scalar("Test/Cosine_Sim", test_cosine_sim)
    writer.add_scalar("Test/MSE", test_mse)
    writer.close()


def main():
    # Example of how you might run multiple experiments:
    epochs_list = [5]
    freeze_vision_options = [True, False]
    freeze_text_options = [True, False]
    lrs = [1e-5]
    hidden_size = 768  # default for FLAVA base
    num_labels = 6     # e.g., 6 emotions

    common_params = {
        "csv_path": "/path/to/your/augmented_data.csv",
        "image_dir": "/path/to/your/image/folder",
        "label_col": "averaged_predictions",
        "text_col": "tweet_text",
        "image_col": "matched_filename",
        "num_labels": num_labels,
        "hidden_size": hidden_size,
    }

    seed = 42

    configs = []
    for epochs in epochs_list:
        for freeze_vis in freeze_vision_options:
            for freeze_txt in freeze_text_options:
                for lr in lrs:
                    log_name = (
                        f"flava_exp_epochs{epochs}_lr{lr}_seed{seed}"
                        + ("_frozen_vision" if freeze_vis else "")
                        + ("_frozen_text" if freeze_txt else "")
                    )
                    config = {
                        "epochs": epochs,
                        "freeze_flava_vision": freeze_vis,
                        "freeze_flava_text": freeze_txt,
                        "log_name": log_name,
                        "learning_rate": lr,
                        **common_params
                    }
                    configs.append(config)

    for config in configs:
        train_and_evaluate(config, seed=seed)


if __name__ == "__main__":
    main()
