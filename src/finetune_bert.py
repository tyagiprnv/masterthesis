import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse

class TextDataset(Dataset):
    def __init__(self, texts, labels, confidences, tokenizer, max_len, label_encoder):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.confidences = confidences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.texts[item],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item_data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long),
        }
        
        if self.confidences is not None:
            item_data['confidences'] = torch.tensor(self.confidences[item], dtype=torch.float)

        return item_data


class ConfidenceWeightedLoss(CrossEntropyLoss):
    def __init__(self, class_weights, no_apply_class_weighting):
        if no_apply_class_weighting:
            super().__init__()
        else:
            super().__init__(weight=class_weights)

    def forward(self, input, target, confidence=None):
        loss = super().forward(input, target)
        
        if confidence is not None:
            weighted_loss = loss * confidence
            return weighted_loss.mean()
        else:
            return loss.mean()

def preprocess_data(label_encoder, apply_preprocessing):
    # Load datasets
    df = pd.read_csv('/work/ptyagi/masterthesis/data/predictions/predictions_cardiffnlp_twitter-roberta-base-emotion-latest.csv')
    text_labels = list(df['label'].unique())
    label_encoder.fit(text_labels)

    replies = pd.read_csv('/work/ptyagi/masterthesis/data/tmp/tweet_replies_feb_2019_en.csv')
    annotations = pd.read_csv('/work/ptyagi/masterthesis/data/test/new_annotation.csv')
    
    if apply_preprocessing:
        min_count = replies['label'].value_counts().min()
        replies = replies.groupby('label').apply(lambda x: x.sample(min_count).reset_index(drop=True)).reset_index(drop=True)

    texts = list(replies['replies'])
    labels = list(df['label'])
    confidences = list(df['score'])

    test_texts = list(annotations['replies'])
    test_labels = list(annotations['manual_label'])

    return texts, labels, confidences, test_texts, test_labels

def main(config_path, mode, apply_preprocessing, save_folder):
    # Load hyperparameters from config file
    with open(config_path) as config_file:
        config = json.load(config_file)
    
    # Initialize label encoder and preprocess data
    label_encoder = LabelEncoder()
    texts, labels, confidences, test_texts, test_labels = preprocess_data(label_encoder, apply_preprocessing)
    
    numerical_labels = label_encoder.transform(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(numerical_labels), y=numerical_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(label_encoder.classes_))

    dataset = TextDataset(texts, labels, confidences, tokenizer, max_len=config['max_len'], label_encoder=label_encoder)
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    test_dataset = TextDataset(test_texts, test_labels, None, tokenizer, max_len=config['max_len'], label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = ConfidenceWeightedLoss(class_weights.to(device), apply_preprocessing)

    if mode == 'train':
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

        for epoch in range(config['epochs']):
            model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                confidences = batch['confidences'].to(device) if 'confidences' in batch else None

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, labels, confidences)
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

            model.eval()
            total_val_loss = 0
            correct_predictions = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch + 1}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    confidences = batch['confidences'].to(device) if 'confidences' in batch else None

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    loss = loss_fn(logits, labels, confidences)
                    total_val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    correct_predictions += (preds == labels).sum().item()

            avg_val_loss = total_val_loss / len(val_loader)
            accuracy = correct_predictions / len(val_dataset)
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        os.makedirs(save_folder, exist_ok=True)
        model.save_pretrained(save_folder)
        tokenizer.save_pretrained(save_folder)
    
    elif mode == 'test':
        model.eval()
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()

        test_accuracy = correct_predictions / len(test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    BASE_PATH = '/work/ptyagi/masterthesis/models/'  
    
    parser = argparse.ArgumentParser(description="Train or Test the BERT Model")
    parser.add_argument('config_path', type=str, help="Path to the config file")
    parser.add_argument('mode', type=str, choices=['train', 'test'], help="Specify mode: 'train' or 'test'")
    parser.add_argument('--preprocess', action='store_true', help="Flag to apply preprocessing on data")
    parser.add_argument('--save_folder', type=str, default='', help="Folder name to save the model within the base path")
    
    args = parser.parse_args()
    
    save_folder = os.path.join(BASE_PATH, args.save_folder)
    
    main(args.config_path, args.mode, args.preprocess, save_folder)