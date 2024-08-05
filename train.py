import argparse
from utils import DataLoader, Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np

# Parser
parser = argparse.ArgumentParser(description='Train sentiment analysis model')
parser.add_argument('--model', type=str, default='klue/roberta-base')
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--warmup_steps', type=int, default=0)
args = parser.parse_args()

base_dir = args.model.replace("/", "-")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Data
X_train, y_train =  DataLoader('data/ratings_train.txt').load_data()
X_test, y_test =  DataLoader('data/ratings_test.txt').load_data()

train_dataset = Tokenizer(tokenizer, X_train, y_train)
test_dataset = Tokenizer(tokenizer, X_test, y_test)

# Model
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

# Trainer
training_args = TrainingArguments(
    output_dir=f'{base_dir}/checkpoints',
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    evaluation_strategy="no",
    save_strategy="epoch",
    metric_for_best_model="accuracy"
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluation
results = trainer.evaluate(test_dataset)
print(results)

model.save_pretrained(f'{base_dir}/model')
