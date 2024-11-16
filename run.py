import pandas as pd
import re
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import gdown
import os
import zipfile
import nltk


nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
morph = MorphAnalyzer()

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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    words = [morph.normal_forms(word)[0] for word in words]
    return ' '.join(words)

def remove_single_letters(text):
    return re.sub(r'\b[а-яё]\b', '', text)

data = pd.read_csv('./payments_main.tsv', sep='\t', header=None)
data.columns = ['id', 'date', 'sum', 'text']

data['text'] = data['text'].apply(preprocess_text)
data['text'] = data['text'].apply(remove_single_letters)

model_path = "./ruBert_tiny"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1gWDHP381W1Sy-TOTLphdap029gsQHyO3/view?usp=sharing"
    output = "rubert_model.zip"
    gdown.download(url, output, quiet=False, fuzzy=True)

    print("Extracting files...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("./")

    os.remove(output)
    print("Model files downloaded and extracted successfully!")

model = AutoModelForSequenceClassification.from_pretrained("./ruBert_tiny", num_labels=9)
tokenizer = AutoTokenizer.from_pretrained("./ruBert_tiny")

device = torch.device('cpu')
model.to(device)

encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=181)
dataset = ResultDataset(encodings)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

predicted_labels = []

model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing batches"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predicted_labels.extend(preds.cpu().numpy())

encoder = {'BANK_SERVICE': 0, 'FOOD_GOODS': 1, 'LEASING': 2, 'LOAN': 3, 'NON_FOOD_GOODS': 4, 'NOT_CLASSIFIED': 5, 'REALE_STATE': 6, 'SERVICE': 7, 'TAX': 8}
decoder = {v: k for k, v in encoder.items()}

decoded_labels = [decoder[label] for label in predicted_labels]

data['label'] = decoded_labels

data.to_csv('submission.tsv', sep='\t', index=False)