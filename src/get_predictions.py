import pandas as pd # type: ignore
import torch # type: ignore
from transformers import pipeline # type: ignore
import multiprocessing
import sys


models = [
    "cardiffnlp/twitter-roberta-base-emotion-latest",  
    "cardiffnlp/twitter-roberta-large-emotion-latest"  
]

def predict_with_model(classification_type, model_name, texts, device):
    print(f"Using model: {model_name}")
    model_pipeline = pipeline(classification_type, model=model_name, device=device)
    if classification_type == "text-classification":
        predictions = model_pipeline(texts, batch_size=32)
    else:
        predictions = []
        for text in data_generator(texts):
            result = model_pipeline(
            text,
            candidate_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"])
            predictions.append(result)

    return predictions

def data_generator(texts):
    for value in texts:
        yield value


def model_worker(classification_type, model_name, texts, device, output_file):
    torch.cuda.set_device(device)
    print(f"device: {device}")
    predictions = predict_with_model(classification_type, model_name, texts, device)

    if classification_type == "text-classification":
        pd.DataFrame(predictions).to_csv(f'/work/ptyagi/masterthesis/data/predictions/{output_file}', index=False)
    else:
        first_labels_scores = [{'label': item['labels'][0], 'score': item['scores'][0]} for item in predictions]
        pd.DataFrame(first_labels_scores).to_csv(f'/work/ptyagi/masterthesis/data/predictions/{output_file}', index=False)


def main(available_devices, csv_file, txt_col ): 

    data = pd.read_csv(csv_file)

    texts = data[txt_col].tolist()  

    processes = []
    num_models = len(models)

    for i in range(num_models):
        model_name = models[i]
        if model_name in ("cardiffnlp/twitter-roberta-large-emotion-latest", "cardiffnlp/twitter-roberta-base-emotion-latest"):
            classification_type = "text-classification"
        else:
            classification_type = "zero-shot-classification"

        device = available_devices[i % len(available_devices)] 
        output_file = f"predictions_{txt_col}_{model_name.replace('/', '_')}.csv"
        p = multiprocessing.Process(target=model_worker, args=(classification_type, model_name, texts, device, output_file))
        processes.append(p)
        p.start()
        print(classification_type)
        print(f'process {p.pid} started')

    for p in processes:
        p.join()
        print(f'process {p.pid} finished')

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
