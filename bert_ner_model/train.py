from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from transformers import DataCollatorForTokenClassification
import re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_metric


MANUAL_NEGATIVE_TECHS = ['Service', 'Foundation', 'Create', 'Control', 'Proven', 'Fluent', 'Orange', 'Qualified', 'Cascade', 'Draft', 'Deputy', 'Promote', 'Faciliate', 'Attend', 'Assess', 'Meet', 'Accept', 'Awareness', 'Gather', 'Written', 'Teamwork', 'adjust', 'Revenue', 'Integrate', 'Humanity', 'Closely', 'Outreach', 'Room', 'Drafted', 'Manage', 'Thanks', 'Good', 'Manager']
TECHS_MATCHING_PATH = 'techs_matching/job_postings_matches.csv'


def clean_tech_list(tech_list_str):
    tech_list = tech_list_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('None', '').replace("'", "").replace('\n', '').split(',')
    return [item.strip() for item in tech_list if item.strip() != '']


def remove_blacklist(tech_list, black_list_techs):
    return set([tech for tech in tech_list if tech not in black_list_techs])


def clean_techs(df):
    techs_dict = {}
    df['techs'] = df['tech_matches_case_sensitivity_True'].apply(clean_tech_list)
    for _, row in df.iterrows():
        for tech in row['techs']:
            techs_dict[tech] = techs_dict.get(tech, 0) + 1

    sorted_tech_dict = dict(sorted(techs_dict.items(), key=lambda item: item[1], reverse=True))
    freq_techs = []
    for key, val in sorted_tech_dict.items():
        if val > 3000:
            freq_techs.append(key)
    freq_techs += MANUAL_NEGATIVE_TECHS

    df['techs'] = df['techs'].apply(remove_blacklist, args=(freq_techs,))

    return df


def set_labels(df):
    all_labels = []
    for _, row in df.iterrows():
        tokens = row['tokens']
        techs = [tech.lower() for tech in row['techs']]
        labels = []
        for token in tokens:
            if token.lower() in techs or np.any([tech in token.lower() for tech in techs]):
                labels.append('PRODUCT')
            else:
                labels.append('NON-PRODUCT')
        all_labels.append(labels)
    df['labels'] = all_labels
    return df


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}



model_checkpoint = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

training_data = pd.read_csv(TECHS_MATCHING_PATH)
training_data = clean_techs(training_data)

training_data['tokens'] = training_data['jobpost'].apply(lambda s: s.split())
training_data = set_labels(training_data)

label_list = ['PRODUCT','NON-PRODUCT']
label_encoding_dict = {'PRODUCT': 1,'NON-PRODUCT': 0}

train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
batch_size = 16

args = TrainingArguments(
    "NER",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

    
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('un-ner.model')
