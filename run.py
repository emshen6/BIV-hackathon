import os
import re
import logging
import zipfile
import configparser

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

import gdown
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode='w')
    ]
)

config = configparser.ConfigParser()
dirname = os.path.dirname(__file__)
config.read(os.path.join(dirname, "config.ini"))

MODEL_URL = config["Model"]["MODEL_URL"]
MODEL_PATH = config["Model"]["MODEL_PATH"]
TOKENIZER_PATH = config["Model"]["TOKENIZER_PATH"]

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
morph = MorphAnalyzer()

class TextClassifierPipeline:
    def __init__(self, model_path, tokenizer_path, labels, max_length=181, batch_size=64, device=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels = labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        if not os.path.exists(self.model_path):
            self._download_model()
        logging.info("Loading model and tokenizer...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=len(self.labels))
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model.to(self.device)
        logging.info("Model and tokenizer loaded successfully.")
    
    def _download_model(self):
        logging.info("Model not found locally. Starting download...")
        url = MODEL_URL
        output = "rubert_model.zip"
        gdown.download(url, output, quiet=False, fuzzy=True)
        logging.info("Extracting model files...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall("./")
        os.remove(output)
        logging.info("Model downloaded and extracted successfully.")
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^а-яё\s]', '', text)
        words = [word for word in text.split() if word not in stop_words]
        words = [morph.normal_forms(word)[0] for word in words]
        return ' '.join(words)

    def remove_single_letters(self, text):
        return re.sub(r'\b[а-яё]\b', '', text)
    
    def preprocess_data(self, data):
        logging.info("Starting text preprocessing...")
        data['text'] = data['text'].apply(self.preprocess_text)
        data['text'] = data['text'].apply(self.remove_single_letters)
        logging.info("Text preprocessing completed.")
        return data
    
    def predict(self, texts):
        logging.info("Tokenizing input data...")
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)
        dataset = ResultDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        logging.info("Starting prediction...")
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        logging.info("Prediction completed.")
        return [self.labels[pred] for pred in predictions]

    def run_pipeline(self, input_path, output_path):
        logging.info("Loading data from %s...", input_path)
        data = pd.read_csv(input_path, sep='\t', header=None, names=['id', 'date', 'sum', 'text'])
        logging.info("Data loaded successfully. Starting pipeline...")
        
        data = self.preprocess_data(data)
        data['label'] = self.predict(data['text'].tolist())
        
        result = data[['id', 'label']]
        logging.info("Saving results to %s...", output_path)
        result.to_csv(output_path, sep='\t', index=False, header=False)
        logging.info("Pipeline completed successfully! Results saved to %s.", output_path)

class ResultDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long)
        }

if __name__ == "__main__":
    LABELS = {0: 'BANK_SERVICE', 1: 'FOOD_GOODS', 2: 'LEASING', 3: 'LOAN', 4: 'NON_FOOD_GOODS', 
              5: 'NOT_CLASSIFIED', 6: 'REALE_STATE', 7: 'SERVICE', 8: 'TAX'}
    pipeline = TextClassifierPipeline(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        labels=LABELS,
        max_length=181,
        batch_size=64,
        device=torch.device('cpu')
    )
    pipeline.run_pipeline(input_path="./payments_main.tsv", output_path="./submission.tsv")
