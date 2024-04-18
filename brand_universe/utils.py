import pandas as pd
import numpy as np
import glob
import ast
from collections import Counter
from llm_label_flow import llm_component

def run_llm_label_flow_gemini_entity(query,data, label_name='llm_label'):
    """
        Runs the LLM label flow on a given dataset.

        Parameters:
        - label_dict (dict): A dictionary mapping the LLM labels to the true labels.
        - query (str): The query to run the LLM on.
        - labels (list): A list of labels to run the LLM on.
        - data (pd.DataFrame): The data to run the LLM on.
        - true_label (str): The column name of the true label.
        - label_name (str): The column name of the LLM label.

        Returns:
        - df: The original dataframe with the LLM labels appended.
        """

    df = data.copy()
    df_dict = {'uid': list(df['uid']), 'text': list(df['text'])}
    predictor = llm_component(query=query, api_type='vertex-api', labels=None, rate_limit=10)
    results = predictor.generate_labels_gemini_entity(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    result_dict = {uid: output for uid, output in zip(results_df['uid'], results_df['output'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x] if x in result_dict.keys() else "[]")
    return df

def ast_literal_eval(x):
    try:
        literal = ast.literal_eval(x)
        return literal
    except:
        return 'cannot_convert'

def llm_result_tag_cleanup(data):
    tags = ['people', 'orgs', 'topics', 'keywords']
    for tag in tags:
        data[f'llm_{tag}'] = data[f'llm_{tag}'].apply(lambda x: "[]" if x == '' else x)
    return data

def clean_up_llm_results(data):
    data = llm_result_tag_cleanup(data)
    tags = ['people', 'orgs', 'topics', 'keywords']
    for tag in tags:
        data[f'llm_{tag}'] = data[f'llm_{tag}'].apply(lambda x: ast_literal_eval(x))
    return data

def obtain_most_common(df):
    # tags = ['people', 'orgs', 'topics', 'keywords']
    tags = ['topics', 'keywords']

    unique_entities = []
    for tag in tags:
        unique_entities += list(df[f'llm_{tag}'])
    unique_entities = [item for item in unique_entities if item!='cannot_convert']
    entities_flattened = [item for sublist in unique_entities for item in sublist]
    entities_flattened = [item.lower() for item in entities_flattened]

    unique_entities_most_common = Counter(entities_flattened).most_common()
    top_common_entities = [x[0] for x in unique_entities_most_common [0:100]]
    return top_common_entities


# Generate Taxonomy
def people_extraction_prompt():
    prompt = "Extract the people discussed in this text and return them as a list of strings like ['people1', 'people2']. If there are no people discussed in the text, return an empty list. Note people could be anyone from celebrities, politicians, fictional characters, tech ceos etc."
    return prompt

def organization_extraction_prompt():
    prompt = "Extract the organizations or companies mentioned in this text and return them as a list of strings like ['organization1','organization2']. If there are no organizations or companies in the text, return an empty list."
    return prompt

def topics_extraction_prompt():
    prompt = "Extract the topics discussed in this text and return them as a list of strings like ['topics1','topics2']."
    return prompt

def keywords_extraction_prompt():
    prompt = "Extract the keywords discussed in this text and return them as a list of strings like ['keyword1','keyword2']. Note kewyords can be unigrams, bigrams, or trigrams."
    return prompt

# Note this method happens from top 100 keywords and topics that are combined together as a set
# from topics and keywords extraction
def keywords_extraction_top_25_prompt():
    prompt = "Extract the top 25 keywords that are related to the given word with an emphasis on top tech or virtual reality keywords. Please return back your response in the format ['keyword1','keyword2']. Note keywords can be unigrams, bigrams, or trigrams. If the word is apple it is the apple company"
    return prompt

def keywords_extraction_top_25_non_tech_explicit_prompt():
    prompt = "Extract the top 25 keywords that are related to the given word. Please return back your response in the format ['keyword1','keyword2']. Note keywords can be unigrams, bigrams, or trigrams. If the word is apple it is the apple company"
    return prompt