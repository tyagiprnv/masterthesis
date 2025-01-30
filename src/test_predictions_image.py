import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_labels, dropout_size, hidden_dim=512):
        super(ImageOnlyClassifier, self).__init__()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_feature_dim = self.clip_model.config.projection_dim  
        
        self.mlp = nn.Sequential(
            nn.Linear(clip_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_features = self.clip_model.get_image_features(image)
        logits = self.mlp(image_features)
        return logits


class ImageDataset(Dataset):
    def __init__(self, data, image_dir, label_col, image_col, clip_processor):
        self.data = data
        self.image_dir = image_dir
        self.label_col = label_col
        self.image_col = image_col
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
        
        return {"image": image, "labels": label_tensor}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_prediction_workflow(
    csv_path,
    image_dir,
    label_col,
    image_col,
    conversation_id_col,
    model_path,
    num_labels,
    batch_size,
    dropout_size,
    output_csv,
    device="cuda"
):
    set_seed(42)
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    data = pd.read_csv(csv_path)
    _, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    _, test_data = train_test_split(temp_data, test_size=1 / 3, random_state=42)
    
    test_dataset = ImageDataset(
        data=test_data,
        image_dir=image_dir,
        label_col=label_col,
        image_col=image_col,
        clip_processor=clip_processor
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ImageOnlyClassifier(
        num_labels=num_labels,
        dropout_size=dropout_size
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    predictions_list = []
    cosine_similarities = []
    kl_divergences = []
    mean_squared_errors = []
    conversation_ids = test_data[conversation_id_col].tolist()
    emotions = ['anger', 'sadness', 'fear', 'joy', 'disgust', 'surprise']
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            
            formatted_predictions = [
                [(emotions[i], prob.item()) for i, prob in enumerate(pred)] for pred in probs
            ]
            predictions_list.extend(formatted_predictions)
            
            batch_cosine_similarities = F.cosine_similarity(probs, labels, dim=1)
            cosine_similarities.extend(batch_cosine_similarities.cpu().numpy())

            batch_kl_divergences = F.kl_div(probs.log(), labels, reduction='none').sum(dim=1)  
            kl_divergences.extend(batch_kl_divergences.cpu().numpy())

            batch_mse = F.mse_loss(probs, labels, reduction='none').mean(dim=1)  
            mean_squared_errors.extend(batch_mse.cpu().numpy())

    print(f"Length of conversation_ids: {len(conversation_ids)}")
    print(f"Length of predictions_list: {len(predictions_list)}")
    print(f"Length of cosine_similarities: {len(cosine_similarities)}")
    print(f"Length of kl_divergences: {len(kl_divergences)}")
    print(f"Length of mean_squared_errors: {len(mean_squared_errors)}")
    result_df = pd.DataFrame({
        "conversation_id": conversation_ids[:len(predictions_list)],
        "predictions": predictions_list,
        "cosine_similarity_model": cosine_similarities,
        "kl_divergence": kl_divergences,
        "mean_squared_error": mean_squared_errors,
    })

    result_df.to_csv(output_csv, index=False)
    print(f"Predictions and evaluation metrics saved to {output_csv}")


image_prediction_workflow(
    csv_path="/work/ptyagi/masterthesis/data/predictions/aug/averaged_predictions.csv",
    image_dir="/work/ptyagi/ClimateVisions/Images/2019/08_August",
    label_col="averaged_predictions",
    image_col="matched_filename",
    conversation_id_col="conversation_id", 
    model_path="/work/ptyagi/masterthesis/src/models/multimodal_experiments_august/exp_adamw_only_clip_lr5e-06_drop0.3_epochs2_seed42/best_model.pt",
    num_labels=6,
    batch_size=16,
    dropout_size=0.3,
    output_csv="/work/ptyagi/masterthesis/data/test_predictions_with_metrics_image.csv",
    device="cuda:5"
)
