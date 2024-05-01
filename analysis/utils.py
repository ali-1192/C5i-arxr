import pandas as pd
import numpy as np
import glob
import os
from constants import sentiment_dict
from llm_label_flow import llm_component
from sklearn.metrics import classification_report, matthews_corrcoef

def run_llm_label_flow_palm(label_dict, query, labels, data, label_name='llm_label'):
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
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=2)
    results = predictor.generate_labels_text_bison_palm(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)

    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    return df

def run_llm_label_flow(label_dict, query, labels, data, label_name='llm_label'):
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
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=20)
    results = predictor.generate_labels_text_bison(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)

    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    return df

def run_llm_label_flow_gemini(label_dict, query, labels, data, label_name='llm_label'):
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
    print(f"Starting gemini for {query} and {labels}")

    df = data.copy()
    df_dict = {'uid': list(df['uid']), 'text': list(df['text'])}
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=10)
    results = predictor.generate_labels_gemini(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)

    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    return df

def obtain_results_gemini_uid_dict(results):
    uid_dict = {}

    for i in range(len(results)):
        try:
            uid_dict.update(ast.literal_eval(results[i]['output']))
        except:
            print(i, results[i]['output'])

    return uid_dict

def run_llm_label_flow_gemini_batch(label_dict, query, labels, data, label_name='llm_label'):
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


    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=1)
    results = predictor.generate_labels_gemini_batch(data=data)
    #
    # results_df = pd.DataFrame.from_dict(results)
    # results_df['label'] = results_df['label'].apply(
    #     lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    # results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)
    #
    # df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    # result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    # df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    return results

def run_llm_label_flow_context(label_dict, query, labels, data, label_name='llm_label'):
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
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=20)
    results = predictor.generate_labels_text_bison_context(data=df_dict)
    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)
    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    result_dict_context = {uid: label for uid, label in zip(results_df['uid'], results_df['context'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    df['context'] = df['uid'].apply(lambda x: result_dict_context[x])
    return df

def run_llm_label_flow_context_comparison(key,label_dict,query, labels, data, true_label, label_name='llm_label'):
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
    results_dict = {}
    results_dict[key] = {}
    df = data.copy()
    df_dict = {'uid': list(df['uid']), 'text': list(df['text'])}
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=20)
    results = predictor.generate_labels_text_bison_context(data=df_dict)
    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)
    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    result_dict_context = {uid: label for uid, label in zip(results_df['uid'], results_df['context'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    df['context'] = df['uid'].apply(lambda x: result_dict_context[x])
    df['true_label'] = df[f'{true_label}']
    df['disagreement'] = df.apply(lambda x: x.true_label != x.llm_label, axis=1)
    mcc = matthews_corrcoef(df['true_label'], df[f'{label_name}'])
    c_r = classification_report(df['true_label'], df[f'{label_name}'])
    results_dict[key]['df'] = df
    results_dict[key]['mcc'] = mcc
    results_dict[key]['classification_report'] = c_r
    results_dict[key]['prompt'] = query
    return results_dict

def run_llm_label_flow_comparison(key,label_dict, query, labels, data, true_label, label_name='llm_label'):
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
    results_dict = {}
    results_dict[key] = {}
    df = data.copy()
    df_dict = {'uid': list(df['uid']), 'text': list(df['text'])}
    predictor = llm_component(query=query, api_type='vertex-api', labels=labels, rate_limit=20)
    results = predictor.generate_labels_text_bison(data=df_dict)

    results_df = pd.DataFrame.from_dict(results)
    results_df['label'] = results_df['label'].apply(
        lambda x: label_dict[x] if x in label_dict.keys() else 'cannot_convert')
    results_df = results_df.loc[results_df['label'] != 'cannot_convert'].reset_index(drop=True)

    df = df.loc[df['uid'].isin(set(results_df['uid']))].reset_index(drop=True)
    result_dict = {uid: label for uid, label in zip(results_df['uid'], results_df['label'])}
    df[f'{label_name}'] = df['uid'].apply(lambda x: result_dict[x])
    # df['true_label'] = df[f'{true_label}'].apply(lambda x: label_dict[x])
    df['true_label'] = df[f'{true_label}']
    df['disagreement'] = df.apply(lambda x: x.true_label!=x.llm_label, axis=1)
    mcc = matthews_corrcoef(df['true_label'], df[f'{label_name}'])
    c_r = classification_report(df['true_label'], df[f'{label_name}'])
    results_dict[key]['df'] = df
    results_dict[key]['mcc'] = mcc
    results_dict[key]['classification_report'] = c_r
    results_dict[key]['prompt'] = query
    return results_dict



def disagreement_percentage(model1_preds, model2_preds, label_dict=False):
    """
    Calculate the percentage of disagreement between two model's predictions.

    Parameters:
    - model1_preds (list of str): Predictions from model 1.
    - model2_preds (list of str): Predictions from model 2.

    Returns:
    - dict: A dictionary containing the breakdown of percentage of disagreements.
    """
    if label_dict:
        model1_preds = [label_dict[x] for x in model1_preds]
        model2_preds = [label_dict[x] for x in model2_preds]

    # Ensure the predictions are of the same length
    if len(model1_preds) != len(model2_preds):
        raise ValueError("The predictions of both models should have the same length.")

    # Initialize a dictionary to store counts of disagreements
    disagreement_counts = {}

    # Loop through paired predictions and count disagreements
    for m1_pred, m2_pred in zip(model1_preds, model2_preds):
        if m1_pred != m2_pred:
            key = (m1_pred, m2_pred)
            disagreement_counts[key] = disagreement_counts.get(key, 0) + 1

    # Convert counts to percentages
    total_disagreements = sum(disagreement_counts.values())
    print(f'The number of disagreements were: {total_disagreements}')
    print(
        f'Out of the {len(model1_preds)} samples the models disagreed: {np.round((total_disagreements / len(model1_preds)) * 100, 2)}% of the time')
    for key, count in disagreement_counts.items():
        disagreement_counts[key] = (count / total_disagreements) * 100

    return disagreement_counts

# Data Preprocessing utils
def load_in_data_manual_export(query):
    """
    Loads in the data for a given query.

    Parameters:
    - query (str): The query to load in the data for.

    Returns:
    - df (pd.DataFrame): The dataframe containing the data for the query.
    """
    dir1 = f'/Users/pvacca/git/project-google-arxr-analytics/data/{query}/*.csv'
    paths = glob.glob(dir1)
    data_list = []
    for path in paths:
        df = pd.read_csv(path,header=10)
        df['tag'] = path.split('/')[-1].split('.')[0].replace(f'{query}_','')
        data_list.append(df)
    df = pd.concat(data_list).reset_index(drop=True)
    df = df.rename(columns={'Url': 'uid', 'Snippet': 'text'})
    df['source'] = 'manual_export'
    return df

def load_in_data_reddit(query):
    dir2 = f'/Users/pvacca/git/project-google-arxr-analytics/data/reddit/{query}_*.json'
    paths = glob.glob(dir2)
    data_list = []
    for path in paths:
        df = pd.read_json(path,lines=True,orient='records')
        df['tag'] = path.split('/')[-1].split('.')[0].replace(f'{query}_','')
        data_list.append(df)
    df = pd.concat(data_list).reset_index(drop=True)
    df = df.rename(columns={'guid': 'uid', 'fullText': 'text'})
    df['source'] = 'reddit'
    return df

def rename_files(tag):
    file_paths = glob.glob(f"../data/{tag}/*.csv")
    for file in file_paths:
        os.system(f" mv {file} {file.replace(tag.title()+'_','').lower()}")

def load_in_data(query):
    manual_export_data = load_in_data_manual_export(query)
    #reddit_data = load_in_data_reddit(query)
    df = pd.concat([manual_export_data]).reset_index(drop=True)
    df['text'] =  df.apply(lambda x: x.Title+x.text if type(x.Title)==str and x['Page Type']=='news' else x.text,axis=1)
    return df

def combine_results_data(all_data,exclude_ids,results_df):
    data_excluded = all_data.loc[all_data['uid'].isin(exclude_ids)].reset_index(drop=True)
    text_map = {text:label for text,label in zip(results_df['text'],results_df['llm_label'])}
    data_excluded['llm_label'] = data_excluded['text'].apply(lambda x: text_map[x] if x in text_map.keys() else None).reset_index(drop=True)
    data_excluded = data_excluded.loc[~data_excluded['llm_label'].isna()].reset_index(drop=True)
    all_results = pd.concat([results_df,data_excluded]).reset_index(drop=True)
    return all_results


def make_preds(query,tag):
    # Load in main data
    main_query_data = load_in_data(query)
    
    # Load in sampled data
    all_data, unique_text_data, excluded_uid_data, train_data, test_data = grab_specific_tag_data_breakdown_test_and_train(main_query_data,tag)
    
    # Condition for sample 
    if len(train_data)>5000:
        train_data = train_data.sample(n=5000,random_state=42)
        
    # Create prompt and make preds
    prompt = generate_prompt(query.title(), tag.title())
    results = run_llm_label_flow_gemini(sentiment_dict, prompt, list(sentiment_dict.keys()), train_data, label_name='llm_label')
    
    print(f'The label distribution for {query} {tag} is: \n with singular results')
    print(results['llm_label'].value_counts()/len(results))
    
    # Combine text to label 
    all_results = combine_results_data(all_data,excluded_uid_data,results)
    
    print(f'The label distribution for {query} {tag} is: \n with all results')
    print(all_results['llm_label'].value_counts()/len(all_results))
    
    # Save data
    all_results.to_json(f'../data/predictions/{query}/{tag}.json',orient='records',lines=True)
    print('Complete')

def make_preds_categories(query,tag):
    # Load in main data
    main_query_data = load_in_data(query)
    
    # Load in sampled data
    all_data, unique_text_data, excluded_uid_data, train_data, test_data = grab_specific_tag_data_breakdown_test_and_train(main_query_data,tag)
    
    # Condition for sample 
    if len(train_data)>5000:
        train_data = train_data.sample(n=5000,random_state=42)
        
    # Create prompt and make preds
    prompt = generate_prompt_categories(tag.title())
    results = run_llm_label_flow_gemini(sentiment_dict, prompt, list(sentiment_dict.keys()), train_data, label_name='llm_label')
    
    print(f'The label distribution for {query} {tag} is: \n with singular results')
    print(results['llm_label'].value_counts()/len(results))
    
    # Combine text to label 
    all_results = combine_results_data(all_data,excluded_uid_data,results)
    
    print(f'The label distribution for {query} {tag} is: \n with all results')
    print(all_results['llm_label'].value_counts()/len(all_results))
    
    # Save data
    all_results.to_json(f'../data/predictions/{query}/{tag}.json',orient='records',lines=True)
    print('Complete')


def grab_specific_tag_data_breakdown_test_and_train(df,tag):
    """
    Grabs a specific tag from a dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe to grab the tag from.
    - tag (str): The tag to grab.

    Returns:
    - df (pd.DataFrame): The dataframe containing only the specified tag.
    """

    df = df.rename(columns={'Url':'uid','Snippet':'text'})

    df = df.loc[df['tag']==tag].reset_index(drop=True)
    df_unique_snippets = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    excluded_uid_from_unique = set(df['uid'])-set(df_unique_snippets['uid'])
    df_test =  df_unique_snippets.sample(n=50,random_state=42).reset_index(drop=True)
    df_test_uids = set(df_test['uid'])
    df_train =  df_unique_snippets.loc[~df_unique_snippets['uid'].isin(df_test_uids)].reset_index(drop=True)
    return df,df_unique_snippets,excluded_uid_from_unique, df_train, df_test

def chunk_dataframe_into_batches(df, batch_size):
    """
    Chunks a dataframe into batches.

    Parameters:
    - df (pd.DataFrame): The dataframe to chunk.
    - batch_size (int): The size of each batch.

    Returns:
    - df_chunks (list of pd.DataFrames): The list of dataframes containing the chunks.
    """
    df_chunks = []
    for i in range(0, len(df), batch_size):
        df_chunks.append(df[i:i + batch_size])
    batches = len(df_chunks)
    df_batched = [{'uids':list(df_chunks[i]['uid']),'texts':list(df_chunks[i]['text']),'batch':i} for i in range(batches)]
    return df_batched


def generate_prompt_batch(query_type, tag, output_format):
    prompt = F"""Hi Gemini! I want you to act as an expert in sentiment! 
    I am trying to get read on what the sentiment is for each post SPECIFIC to {tag.title()} {query_type}. 
    I.e. is the sentiment, positive, negative, or neutral TOWARDS {tag.title()} {query_type}.
    Think through your answer, and remind yourself is this SPECIFICALLY POSITIVE, NEGATIVE, or NEUTRAL towards {tag.title()} {query_type}, or is this tone just
    positive,negative,neutral in general.
    My life depends on getting the entity-specifc answer correct! 
    Again when labeling the sentiment label it under the condition of its SPECIFIC sentiment towards {tag.title()} {query_type}
    \n Return the labels in the form: "{output_format}" in the order of the posts 

    Posts:"""
    return prompt

def generate_prompt(query_type, tag):
    prompt = f"""Hi Gemini! I want you to act as an expert in sentiment! 
    I am trying to get read on what the sentiment is for each post SPECIFIC to {tag.title()} {query_type}. 
    I.e. is the sentiment, positive, negative, or neutral TOWARDS {tag.title()} {query_type}.
    Think through your answer, and remind yourself is this SPECIFICALLY POSITIVE, NEGATIVE, or NEUTRAL towards {tag.title()} {query_type}, or is this tone just
    positive,negative,neutral in general.
    My life depends on getting the entity-specifc answer correct! 
    Again when labeling the sentiment label it under the condition of its SPECIFIC sentiment towards {tag.title()} {query_type}
    \n Output: -negative -positive -neutral"""
    return prompt

def generate_prompt_categories(tag):
    prompt = F"""Hi Gemini! I want you to act as an expert in sentiment! 
    I am trying to get read on what the sentiment is for each post SPECIFIC to the company {tag.title()} in the Augmented Reality (AR), Virtual Reality (VR), and Extended Reality (XR) field. 
    I.e. is the sentiment, positive, negative, or neutral TOWARDS {tag.title()} {tag.title()} in the Augmented Reality (AR), Virtual Reality (VR), and Extended Reality (XR) field.
    Think through your answer, and remind yourself is this SPECIFICALLY POSITIVE, NEGATIVE, or NEUTRAL towards {tag.title()} {tag.title()} in the Augmented Reality (AR), Virtual Reality (VR), and Extended Reality (XR) field, or is this tone just
    positive,negative,neutral in general.
    My life depends on getting the entity-specifc answer correct! 
    Again when labeling the sentiment label it under the condition of its SPECIFIC sentiment towards {tag.title()} {tag.title()} in the Augmented Reality (AR), Virtual Reality (VR), and Extended Reality (XR) field.
    \n Output: -negative -positive -neutral"""
    return prompt

