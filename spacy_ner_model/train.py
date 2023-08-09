import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
import pandas as pd

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


def find_techs_indices(row):
    techs = [tech.lower() for tech in row['techs']]
    job_post = row['jobpost'].lower()
    indices_list = []
    for tech in techs:
        start_index = job_post.find(tech)
        end_index = start_index + len(tech)
        indices_list.append((start_index, end_index))
    return indices_list


def preprocess_data(df):
    data = clean_techs(df)
    data['techs_annotations'] = data.apply(find_techs_indices, axis=1)
    training_data = []
    for _, example in data.iterrows():
        temp_dict = {}
        temp_dict['text'] = example['jobpost']
        temp_dict['entities'] = []
        for annotation in example['techs_annotations']:
            start = annotation[0]
            end = annotation[1]
            label = 'PRODUCT'
            temp_dict['entities'].append((start, end, label))
        training_data.append(temp_dict)

    return training_data


data = pd.read_csv(TECHS_MATCHING_PATH)
training_data = preprocess_data(data)

nlp = spacy.load("en_core_web_sm")

doc_bin = DocBin()
for training_example in tqdm(training_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels: 
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("spacy_ner_model/train.spacy")
