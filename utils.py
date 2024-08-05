import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DataLoader:
    """
        Loads training and test data from a delimited text file into memory

        Args:
            filepath (str): The path to the data file.
    """

    def __init__(self, filepath, delimiter='\t'):
        self.filepath = filepath
        self.delimiter = delimiter

    def load_data(self):
        """
            Returns: A list of documents (X) and a list of corresponding labels (y)
        """
        df = pd.read_csv(self.filepath, delimiter=self.delimiter)
        X = df['document'].astype(str).tolist()
        y = df['label'].tolist()
        return X, y
    
class Tokenizer:
    """
        Tokenizes the text using a provided tokenizer.

        Args:
            tokenizer: Huggingface tokenizer.
            sentences (list): List of sentences to tokenize.
            labels (list): List of labels corresponding to the sentences.
    """
    def __init__(self, tokenizer, sentences, labels, max_length=128):
        self.encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class Predictor:
    """
        Predicts the sentiment of given piece of text using a finetuned model.

        Args:
            tokenizer: Huggingface tokenizer.
            model_path (str): Path to the finetuned model.
    """
    def __init__(self, tokenizer, model_path, num_labels=2):
       
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.model.eval()

    def predict_sentiment(self, text):
        """
            Returns: Dictionary of predicted sentiment label and confidence score
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        pred_class= probabilities.argmax().item()
        pred_score = probabilities.max().item()

        return {'label': pred_class, 'score': f"{pred_score:.4f}"}