# Job Posting Technologies Extraction Project

## Overview

This project aims to extract relevant technologies mentioned in job postings using Natural Language Processing (NLP) techniques. The project utilizes the 'Online Job Posting' dataset from Kaggle, which contains a collection of job postings. The goal is to develop a model that can identify and extract technologies mentioned in each job posting.

## Dataset

The dataset used for this project is the 'Online Job Posting' dataset from Kaggle, which can be found [here](https://www.kaggle.com/datasets/madhab/jobposts). The dataset contains a collection of job postings, each with various attributes such as job title, company, job description, and more.

## Methodology

1. **Data Preprocessing**: The job postings data is cleaned and preprocessed to remove any irrelevant information and ensure consistent formatting. This includes lowercasing, removing special characters, and tokenizing the text.

2. **Technology List**: A list of relevant technologies (keywords) is scraped from Slintel website. This list includes programming languages, tools, frameworks, and other technological terms commonly found in job postings.

3. **Keyword Matching**: In the absence of a 'gold dataset', the project creates a baseline training dataset using keyword matching. Job postings are matched against Slintel's technologies scraped mapping, allowing for the identification of relevant technologies mentioned in the postings.

4. **Matcher Preprocess**: To improve the precision of the keyword matcher, a two-step cleanup process is applied:
   a. **TF-IDF**: Term Frequency-Inverse Document Frequency (TF-IDF) is used to filter out irrelevant technologies from the matcher's results.
   b. **POS Tagging**: Part-of-Speech (POS) tagging is used to identify and retain only the relevant keywords in the matcher's output.

5. **Training Phase**: Different models are employed for the training phase using the baseline training dataset:
   a. **Finetuned BERT**: A BERT-based model is finetuned on the baseline training dataset to improve the accuracy of technology extraction.
   b. **Finetuned spaCy's NER Model**: spaCy's Named Entity Recognition (NER) model is trained on the dataset to recognize and extract relevant technology entities.

6. **Evaluation**: 100 Job postings were manually tagged in order to create a datset for tests and evaluations. The model's performance is evaluated by comparing the extracted technologies with the test dataset. Precision and recall metrics are calculated to assess the model's accuracy in identifying relevant technologies.

7. **Output Column**: A new column is added to the dataset, which lists all the relevant technologies identified in each job posting.

## Usage

1. **Environment Setup**: Make sure you have the required libraries installed. You can download the job posting data (from any source), run each model separately, and analyse the results.
