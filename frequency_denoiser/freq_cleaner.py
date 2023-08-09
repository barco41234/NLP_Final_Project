import pandas as pd

MANUAL_NEGATIVE_TECHS = ['Service', 'Foundation', 'Create', 'Control', 'Proven', 'Fluent', 'Orange', 'Qualified', 'Cascade', 'Draft', 'Deputy', 'Promote', 'Faciliate', 'Attend', 'Assess', 'Meet', 'Accept', 'Awareness', 'Gather', 'Written', 'Teamwork', 'adjust', 'Revenue', 'Integrate', 'Humanity', 'Closely', 'Outreach', 'Room', 'Drafted', 'Manage', 'Thanks', 'Good', 'Manager']
TECHS_MATCHING_PATH = 'techs_matching/job_postings_matches.csv'


def clean_tech_list(tech_list_str):
    tech_list = tech_list_str.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('None', '').replace("'", "").replace('\n', '').split(',')
    return [item.strip() for item in tech_list if item.strip() != '']

def remove_blacklist(tech_list, black_list_techs):
    return set([tech for tech in tech_list if tech not in black_list_techs])


def predict(df):
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

    df['clean_techs'] = df['techs'].apply(remove_blacklist, args=(freq_techs,))

    return df



if __name__ == "__main__":
    job_posts_df = pd.read_csv(TECHS_MATCHING_PATH)
    job_posting_inference = predict(job_posts_df)

    job_posting_inference.to_csv(f"frequency_denoiser/job_postings_frequency_cleaner.csv")
    job_posting_inference[:100].to_csv(f"frequency_denoiser/job_postings_frequency_cleaner_100samples.csv")


