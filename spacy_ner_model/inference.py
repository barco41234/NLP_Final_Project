import spacy
import pandas as pd

MODEL_PATH = 'spacy_ner_model/so-for-best-model/model-last'
JOB_POSTING_PATH = "techs_matching/job_postings_matches.csv"

model = spacy.load(MODEL_PATH)


def infer_job_post(job_post):
    doc = model(job_post)
    techs = []
    for ent in doc.ents:
        if ent.label_ == "PRODUCT":
            techs.append(ent.text)
    return set(techs)


def predict(df):
    df['spacy_infer_techs'] = df['jobpost'].apply(infer_job_post)
    return df



if __name__ == "__main__":
    job_postings_df = pd.read_csv(JOB_POSTING_PATH)
    job_postings_df_inference = predict(job_postings_df)
    job_postings_df_inference.to_csv(f"spacy_ner_model/job_postings_spacy_inference.csv")
    job_postings_df_inference[:100].to_csv(f"spacy_ner_model/job_postings_spacy_inference_100samples.csv")
