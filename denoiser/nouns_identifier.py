import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

JOB_POSTING_PATH = "techs_matching/job_postings_matches.csv"


def parse_list_of_strings(string_list):
    return re.findall(r"'([^']+)'", string_list)


def find_intersection(nouns, techs):
    # Convert the lists to sets to find the intersection
    nouns_set = set(nouns)
    techs_lc = [t.lower() for t in techs]
    techs_set = set(techs_lc)

    # Find the intersection between the sets
    intersection_result = nouns_set.intersection(techs_set)

    return intersection_result



def extract_nouns(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Filter out stopwords (common words that don't carry much meaning)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Use POS tagging to identify the parts of speech for each word
    tagged_words = pos_tag(words)

    # Extract nouns (NN, NNS, NNP, NNPS) from the tagged words
    nouns = [word for word, pos in tagged_words if not pos.startswith('V')]

    return nouns

def predict(job_posts_df):
    job_posts_df['tech_matches_case_sensitivity_False'] = job_posts_df['tech_matches_case_sensitivity_False'].apply(parse_list_of_strings)

    job_posts_df['nouns'] = job_posts_df['jobpost'].apply(extract_nouns)
    job_posts_df['noun_techs'] = job_posts_df.apply(lambda row: find_intersection(row['nouns'], row['tech_matches_case_sensitivity_False']), axis=1)

    return job_posts_df

if __name__ == "__main__":
    job_posts_df = pd.read_csv(JOB_POSTING_PATH)
    job_posting_inference = predict(job_posts_df)

    job_posts_df.to_csv(f"denoiser/job_postings_noun_matches.csv")
    job_posts_df[:100].to_csv(f"denoiser/job_postings_noun_matches_100samples.csv")




