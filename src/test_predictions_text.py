import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

class TextClassifier(nn.Module):
    def __init__(self, num_labels, txt_model, dropout_size, hidden_dim=512):
        super(TextClassifier, self).__init__()
        
        self.roberta = AutoModel.from_pretrained(txt_model)
        text_feature_dim = self.roberta.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_size),
            nn.Linear(hidden_dim, num_labels)
        )

    def freeze_roberta(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        text_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] 
        logits = self.mlp(text_features)
        return logits

class TextDataset(Dataset):
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

def text_prediction_workflow(
    csv_path,
    label_col,
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

    roberta_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data = pd.read_csv(csv_path)
    _, temp_data = train_test_split(data, test_size=0.3, random_state=seed)
    _, test_data = train_test_split(temp_data, test_size=1 / 3, random_state=seed)

    test_dataset = TextDataset(
        data=test_data,
        label_col=label_col,
        text_col=text_col,
        roberta_tokenizer=roberta_tokenizer,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = TextClassifier(
        num_labels=num_labels,
        txt_model=tokenizer_name,
        dropout_size=dropout_size,
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
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

    result_df = pd.DataFrame({
        "conversation_id": conversation_ids[:len(predictions_list)],
        "predictions": predictions_list,
        "cosine_similarity_model": cosine_similarities,
        "kl_divergence": kl_divergences,
        "mean_squared_error": mean_squared_errors,
    })

    result_df.to_csv(output_csv, index=False)
    print(f"Predictions and evaluation metrics saved to {output_csv}")

text_prediction_workflow(
    csv_path="/work/ptyagi/masterthesis/data/predictions/aug/averaged_predictions.csv",
    label_col="averaged_predictions",
    text_col="tweet_text",
    conversation_id_col="conversation_id", 
    model_path="/work/ptyagi/masterthesis/src/models/multimodal_experiments_august/exp_only_roberta_large_lr5e-06_drop0.3_epochs5_seed42/best_model.pt",
    tokenizer_name="cardiffnlp/twitter-roberta-large-emotion-latest",
    num_labels=6,
    batch_size=16,
    dropout_size=0.3,
    output_csv="/work/ptyagi/masterthesis/data/test_predictions_with_metrics_text.csv",
    device="cuda:7",
)
