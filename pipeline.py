import argparse
from utils import Predictor

# Parser
parser = argparse.ArgumentParser(description='Predict sentiment of a single input sentence')
parser.add_argument('--model', type=str, default='klue/roberta-base')
parser.add_argument('--input', type=str, required=True, help='Single sentence or word')
args = parser.parse_args()

base_dir = args.model.replace("/", "-")

# Predictor
predictor = Predictor(tokenizer=(args.model), model_path=f'{base_dir}/model')

prediction = predictor.predict_sentiment(args.input)

labels = {0: 'Negative', 1: 'Positive'}

output = [{'input': args.input, 'label': labels[prediction['label']], 'score': prediction['score']}]

print(output)