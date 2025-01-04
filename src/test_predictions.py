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
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


class MultiModalClassifier(nn.Module):
    def __init__(self, num_labels, txt_model, dropout_size, hidden_dim=512):
        super(MultiModalClassifier, self).__init__()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = AutoModel.from_pretrained(txt_model)

        clip_feature_dim = self.clip_model.config.projection_dim  
        text_feature_dim = self.text_model.config.hidden_size
        combined_dim = clip_feature_dim + text_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(hidden_dim, num_labels)
        )

    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def freeze_roberta(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, image, input_ids, attention_mask):
        image_features = self.clip_model.get_image_features(image)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] 

        combined_features = torch.cat([image_features, text_features], dim=-1)

        logits = self.mlp(combined_features)
        return logits


class MultiModalDataset(Dataset):
    def __init__(self, data=None, csv_path=None, image_dir=None, label_col=None, image_col=None, text_col=None, clip_processor=None, roberta_tokenizer=None, max_length=512):
        if data is not None:
            self.data = data
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Either `data` or `csv_path` must be provided.")

        self.image_dir = image_dir
        self.label_col = label_col
        self.image_col = image_col
        self.text_col = text_col
        self.clip_processor = clip_processor
        self.tokenizer = roberta_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx][self.image_col]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        text = self.data.iloc[idx][self.text_col]
        text_inputs = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        label_tuples = eval(self.data.iloc[idx][self.label_col])  
        label_probs = [prob for _, prob in label_tuples]  
        label_tensor = torch.tensor(label_probs, dtype=torch.float)

        return {
            "image": image,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": label_tensor,
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multimodal_prediction_workflow(
    csv_path,
    image_dir,
    label_col,
    image_col,
    text_col,
    conversation_id_col,
    model_path,
    tokenizer_name,
    num_labels,
    batch_size,
    dropout_size,
    output_csv,
    device="cuda"
):

    seed = 42
    set_seed(seed)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    roberta_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data = pd.read_csv(csv_path)
    _, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    _, test_data = train_test_split(temp_data, test_size=1 / 3, random_state=seed)

    test_dataset = MultiModalDataset(
        data=test_data,
        image_dir=image_dir,
        label_col=label_col,
        image_col=image_col,
        text_col=text_col,
        clip_processor=clip_processor,
        roberta_tokenizer=roberta_tokenizer,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MultiModalClassifier(
        num_labels=num_labels,
        txt_model=tokenizer_name,
        dropout_size=dropout_size,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions_list = []
    cosine_similarities = []
    conversation_ids = test_data[conversation_id_col].tolist()
    emotions = ['anger', 'sadness', 'fear', 'joy', 'disgust', 'surprise']

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)

            formatted_predictions = [
                [(emotions[i], prob.item()) for i, prob in enumerate(pred)] for pred in probs
            ]
            predictions_list.extend(formatted_predictions)

            batch_cosine_similarities = F.cosine_similarity(probs, labels, dim=1)
            cosine_similarities.extend(batch_cosine_similarities.cpu().numpy())

    result_df = pd.DataFrame({
        "conversation_id": conversation_ids[:len(predictions_list)],
        "predictions": predictions_list,
        "cosine_similarity_model": cosine_similarities,
    })

    result_df.to_csv(output_csv, index=False)
    print(f"Predictions and cosine similarities saved to {output_csv}")


multimodal_prediction_workflow(
    csv_path="/work/ptyagi/masterthesis/data/predictions/aug/averaged_predictions.csv",
    image_dir="/work/ptyagi/ClimateVisions/Images/2019/08_August",
    label_col="averaged_predictions",
    image_col="matched_filename",
    text_col="tweet_text",
    conversation_id_col="conversation_id", 
    model_path="/work/ptyagi/masterthesis/src/models/multimodal_experiments_august/exp_roberta_base_lr1e-05_drop0.3_epochs2_seed42/best_model.pt",
    tokenizer_name="cardiffnlp/twitter-roberta-base-emotion-latest",
    num_labels=6,
    batch_size=8,
    dropout_size=0.3,
    output_csv="/work/ptyagi/masterthesis/data/test_predictions_with_cosine.csv",
    device="cuda:7",
)



