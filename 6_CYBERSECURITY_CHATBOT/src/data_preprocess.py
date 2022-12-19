import json
import os
import pandas as pd
import torch

def df_to_list(df):
    """Convert a dataframe to a list of lists.
    Args:
        df (pandas.DataFrame): The dataframe to convert.
    Returns:
        list: A list of list.
    """
    # convert the DataFrame to a list of records (tuples)
    records = df.to_records(index=False)

    # extract the questions and answers from the tuples and create a list of tuples
    tuples = [[record[0], record[1]] for record in records]

    return tuples

# Define the evaluation metric as the accuracy
def calculate_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=2)
    correct = torch.eq(predictions, targets).float()
    return correct.sum() / correct.numel()