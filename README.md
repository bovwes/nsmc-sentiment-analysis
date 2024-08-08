# NSMC Sentiment Analysis

This is a script to perform fine-tuning and inference on pre-trained Korean language models using the NSMC dataset.

## Data

The [Naver Sentiment Movie Corpus (NSMC)](https://github.com/e9t/nsmc) consists of 200K annotated movie reviews. This dataset can be used to finetune pre-trained language models for Korean sentiment analysis.

- `ratings_train.txt`: 150K reviews for training
- `ratings_test.txt`: 50K reviews held out for testing

## Dependencies

You can install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Also install the CUDA or CPU version of PyTorch 2.4.0 from [here](https://pytorch.org/get-started/locally/).

## Train

You can train and evaluate a model on the NSMC corpus directly from Huggingface using the following command, where `model_name` links to any Huggingface model.

```bash
python train.py --model [model_name]
```

**Training arguments:**

| Parameter            | Type  | Default value       | Description                                |
| -------------------- | ----- | ------------------- | ------------------------------------------ |
| `--model`            | str   | "klue/roberta-base" | Name of the model, as found on Huggingface |
| `--num_train_epochs` | int   | 3                   | Number of training epochs                  |
| `--train_batch_size` | int   | 64                  | Batch size for training                    |
| `--eval_batch_size`  | int   | 64                  | Batch size for evaluation                  |
| `--learning-rate`    | float | 5e-5                | Learning rate                              |
| `--weight_decay`     | float | 0                   | Weight decay                               |
| `--warmup_steps`     | int   | 0                   | Warmup steps                               |

## Predict

After you have trained a model, you can use that model to predict whether a piece of text has a negative or postive sentiment. Provide a `.txt` file consisting of one text sample per line:

```bash
python predict.py --model [model_name] --input-path [path] --output-path [path]
```

**Example output:**

```
document	label	score
재미지다!	1	0.9554
재밋어~~ 천정명 나오니깐 봄♥♥♥♥	1	0.9829
수호천사 케빈 스케이시~ 좀 잘 만들지....	0	0.5127
3류 막장 야쿠자극화,	0	0.9987
난 왜 지루하게만 느껴지지?	0	0.9948
```

Alternatively, you can use `pipeline.py` to predict the sentiment of a single word or sentence:

```bash
python pipeline.py --model [model_name] --input ["word_or_sentence"]
```

**Example output:**

```json
[{ "input": "노잼", "label": "Negative", "score": "0.9987" }]
```

## Results

Below are some pre-trained language models and their performance on NSMC. Each model was fine-tuned on the `ratings_train` set and evaluated on the `ratings_test` set. Batch sizes were set to 64 and each model was trained for 3 epochs, starting at a learning rate of 5e-5.

| Model                        | Accuracy (%) |
| ---------------------------- | ------------ |
| KLUE-RoBERTa-small           | 90.16        |
| KR-BERT character            | 90.03        |
| BERT base multilingual cased | 87.18        |
