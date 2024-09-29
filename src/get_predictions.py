import pandas as pd
import torch
from transformers import pipeline
import multiprocessing
import sys


models = [
    "cardiffnlp/twitter-roberta-base-emotion-latest",  
    "cardiffnlp/twitter-roberta-large-emotion-latest",  
    "facebook/bart-large-mnli", 
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" 
]

def predict_with_model(classification_type, model_name, texts):
    print(f"Using model: {model_name}")
    model_pipeline = pipeline(classification_type, model=model_name)
    predictions = model_pipeline(texts)
    return predictions

def model_worker(model_name, texts, device):
    torch.cuda.set_device(device)
    return predict_with_model(model_name, texts)

def main(available_devices, csv_file, txt_col ): 

    data = pd.read_csv(csv_file)

    texts = data[txt_col].tolist()  

    processes = []
    num_models = len(models)

    for i in range(num_models):
        model_name = models[i]
        device = available_devices[i % len(available_devices)] 

        p = multiprocessing.Process(target=model_worker, args=(model_name, texts, device))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All model predictions completed.")

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python script.py <file_path> <text_column> <device1> <device2> ...")
        sys.exit(1)

    file_path = sys.argv[1]  
    text_column = sys.argv[2] 
    available_devices = list(map(int, sys.argv[3:]))  
    print(f"Using file: {file_path}, text column: {text_column}, available devices: {available_devices}")
    
    multiprocessing.set_start_method('spawn')  
    main(available_devices, file_path, text_column)
