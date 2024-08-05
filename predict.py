import argparse
import csv
from utils import Predictor

# Parser
parser = argparse.ArgumentParser(description='Predict sentiment of multiple input sentences from a file')
parser.add_argument('--model', type=str, default='klue/roberta-base')
parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-path', type=str, default='out.txt')
args = parser.parse_args()

base_dir = args.model.replace("/", "-")

# Predictor
predictor = Predictor(tokenizer=(args.model), model_path=f'{base_dir}/model')

predicitons = []

# Read input file
i = 0
with open(args.input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        i += 1
        if line:
            print(f"Processing line {i}: {line}", end="\r")
            prediction = predictor.predict_sentiment(line)
            predicitons.append({'document': line, 'label': prediction['label'], 'score': prediction['score']})
        else:
            print(f"Skipping empty line {i}")

# Write output file
with open(args.output_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['document', 'label', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for prediction in predicitons:
        writer.writerow(prediction)

print("")
print(f"Finished. Results saved to {args.output_path}")
