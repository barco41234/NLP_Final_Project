from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import pandas as pd

NER_MODEL_PATH = './un-ner.model/'
JOB_POSTING_PATH = "techs_matching/job_postings_matches.csv"

label_list = ['NON-PRODUCT', 'PRODUCT']
label_encoding_dict = {'PRODUCT': 1, 'NON-PRODUCT': 0}
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH, num_labels=len(label_list))


def infer_job_post(paragraph):
    tokens = tokenizer(paragraph)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [label_list[i] for i in predictions]

    words = tokenizer.batch_decode(tokens['input_ids'])
    techs = []
    for i, p in enumerate(predictions):
        if p == 'PRODUCT':
            techs.append(words[i])
    
    return set(techs)


def find_index(text):
    keywords = ['qualifications', 'requirements']
    for keyword in keywords:
        index = text.find(keyword)
        if index != -1:
            return index
    return -1


def cut_job_posts(df):
    df['jobpost'] = df['jobpost'].apply(lambda j: j.lower())
    df['indices'] = df['jobpost'].apply(find_index)

    # Remove the prefix up to the first occurrence of 'qualifications' or 'requirements' in each row
    df['jobpost_cut'] = df.apply(lambda row: row['jobpost'][row['indices']+1:row['indices']+2300], axis=1)

    return df


def predict(df):
    df = cut_job_posts(df)
    df['infer_techs'] = df['jobpost_cut'].apply(infer_job_post)
    return df


if __name__ == "__main__":
    job_postings_df = pd.read_csv(JOB_POSTING_PATH)
    job_postings_df = cut_job_posts(job_postings_df)
    job_postings_df_inference = predict(job_postings_df)