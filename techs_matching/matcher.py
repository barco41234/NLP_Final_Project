import pandas as pd
import re
from flashtext import KeywordProcessor

SLINTEL_TECHS_MAPPING_PATH = "techs_matching/slintel_techs_full_mapping.csv"
JOB_POSTING_PATH = "data job posts.csv"


def parse_list_of_strings(string_list):
    return re.findall(r"'([^']+)'", string_list)


def find_matches(job_posting, kp):
    return set(kp.extract_keywords(job_posting))


def create_tech_set(path = SLINTEL_TECHS_MAPPING_PATH):
    techs_mapping = pd.read_csv(path)
    techs_mapping['tech_lists'] = techs_mapping['techs'].apply(parse_list_of_strings)
    techs = techs_mapping['tech_lists']
    tech_sets = [set(lst) for lst in techs]
    all_techs = set().union(*tech_sets)
    all_techs.remove(" ") # Remove space tech
    return all_techs


def job_posts_find_matches(job_posts_df, kp, case_sensitivity):
    col_name = f"tech_matches_case_sensitivity_{case_sensitivity}"
    job_posts_df[col_name] = job_posts_df['jobpost'].apply(find_matches, args=(kp,))
    return job_posts_df
    


def predict(job_posts_df):
    techs = create_tech_set()

    for case_sensitivity in [True, False]:
        keyword_processor = KeywordProcessor(case_sensitive=case_sensitivity)
        for tech in techs:
            keyword_processor.add_keyword(tech)

        job_posts_matches = job_posts_find_matches(job_posts_df, keyword_processor, case_sensitivity)

    return job_posts_matches

if __name__ == "__main__":
    job_posts_df = pd.read_csv(JOB_POSTING_PATH)
    job_posting_inference = predict(job_posts_df)
    job_posting_inference.to_csv(f"techs_matching/job_postings_matches.csv")

    job_posting_inference[:100].to_csv(f"techs_matching/job_postings_100sample_matches.csv")
