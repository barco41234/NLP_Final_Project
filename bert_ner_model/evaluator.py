import pandas as pd
from inference import predict
import numpy as np

TAGGED_DATA_PATH = 'techs_matching/test_inference_predict.csv'


def clean_tech_list(tech_list_str):
    tech_list = tech_list_str.replace('[', '').replace(']', '').replace('None', '').replace("'", "").replace('\n', '').split(',')
    return [item.strip() for item in tech_list if item.strip() != '']


def find_union_set(row):
    techs_from_mapping = clean_tech_list(row['Techs_From_Mapping'])
    techs_not_from_mapping = clean_tech_list(row['Techs_Not_From_Mapping'])

    return set(techs_from_mapping) | set(techs_not_from_mapping)


def find_intersection(row):
    all_techs_lower = set([tech.lower() for tech in row['all_techs']])
    noun_techs_lower = set([tech.lower() for tech in row['infer_techs']])

    return all_techs_lower & noun_techs_lower


def find_unique_techs(row, col_A, col_B):
    return row[col_A] - row[col_B]


def count_techs(row):
    return len(row)


def evaluate_model(test_data_path):
    test_df = pd.read_csv(test_data_path)
    test_inference = predict(test_df)
    test_inference['all_techs'] = test_inference.apply(find_union_set, axis=1)
    test_inference['all_techs_n'] = test_inference['all_techs'].apply(count_techs)

    test_inference['TP'] = test_inference.apply(find_intersection, axis=1)
    test_inference['TP_N'] = test_inference['TP'].apply(count_techs)
    
    test_inference['FN'] = test_inference.apply(find_unique_techs, args=('all_techs', 'infer_techs'), axis=1)
    test_inference['FN_N'] = test_inference['FN'].apply(count_techs)

    test_inference['FP'] = test_inference.apply(find_unique_techs, args=('infer_techs', 'all_techs'), axis=1)
    test_inference['FP_N'] = test_inference['FP'].apply(count_techs)

    test_inference['precision'] = test_inference['TP_N'] / (test_inference['TP_N'] + test_inference['FP_N'])
    test_inference['recall'] = test_inference['TP_N'] / (test_inference['TP_N'] + test_inference['FN_N'])

    mean_precision = np.mean(test_inference['precision'])
    mean_recall = np.mean(test_inference['recall'])
    non_zero_predictions = len(test_inference[test_inference['TP_N'] > 0])
    print(f'Avg Precision: {mean_precision}')
    print(f'Avg Recall: {mean_recall}')
    print(f'Non-zero predictions: {non_zero_predictions}')
    print(f'Non-zero predictions: {non_zero_predictions / len(test_inference) * 100} %')

    test_inference.to_csv('bert_ner_model/test_inference_evaluation.csv')


if __name__ == "__main__":
    evaluate_model(TAGGED_DATA_PATH)