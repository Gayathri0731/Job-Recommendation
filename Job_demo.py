import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_data = pd.read_csv('jobdataset.csv')
sel_features = ['Company Name','Job Description','Location','Position','Required Courses']
combined_features = job_data['Company Name']+' '+job_data['Job Description']+' '+ job_data['Location']+' '+job_data['Position']+' '+job_data['Required Courses']
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity=cosine_similarity(feature_vector)

import difflib

job_name = input('Enter job Title: ')

list_of_all_titles = job_data['Job Title'].tolist()

find_close_match = difflib.get_close_matches(job_name, list_of_all_titles)

if find_close_match:
    close_match = find_close_match[0]

    index_of_job = job_data[job_data['Job Title'] == close_match]['Index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_job]))

    sorted_similar_job = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('Job Recommendations:\n')

    i = 1
    for jobs in sorted_similar_job:
        index = jobs[0]
        job_row = job_data.iloc[index]

        if i < 11:
            print(f"{i}. {job_row['Job Title']} - {job_row}")
            i += 1
else:
    print("No close matches found.")
