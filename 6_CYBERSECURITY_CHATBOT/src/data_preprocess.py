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
  """function to calculate accuracy of model
  Args:
      logits (predictions): model predictions 
      targets (labels): correct labels

  Returns:
      float: accuracy of model
  """    
  predictions = torch.argmax(logits, dim=2)
  correct = torch.eq(predictions, targets).float()
  return correct.sum() / correct.numel()

def align_tensors(input_tensors, target_tensors):
  # Find the maximum length of the input and target tensors
  max_input_len = max([input_tensor.size(1) for input_tensor in input_tensors])
  max_target_len = max([target_tensor.size(1) for target_tensor in target_tensors])

  # Align the input tensors by padding to the maximum length
  aligned_input_tensors = []
  for input_tensor in input_tensors:
    padding = torch.zeros(input_tensor.size(0), max_input_len - input_tensor.size(1), dtype=torch.long)
    aligned_input_tensor = torch.cat((input_tensor, padding), dim=1)
    aligned_input_tensors.append(aligned_input_tensor)
  
  # Align the target tensors by padding to the maximum length
  aligned_target_tensors = []
  for target_tensor in target_tensors:
    padding = torch.zeros(target_tensor.size(0), max_target_len - target_tensor.size(1), dtype=torch.long)
    aligned_target_tensor = torch.cat((target_tensor, padding), dim=1)
    aligned_target_tensors.append(aligned_target_tensor)

  # Stack the aligned input and target tensors into a single tensor
  stacked_input_tensor = torch.stack(aligned_input_tensors)
  stacked_target_tensor = torch.stack(aligned_target_tensors)

  return stacked_input_tensor, stacked_target_tensor

# find maximum words in dataframe answers
def max_words(df):
  """function to find maximum words in dataframe answers
  Args:
      df (dataframe): dataframe with answers
  Returns:
      int: maximum words in answers
  """    
  max_words = 0
  for i in range(len(df)):
    if len(df.iloc[i, 1].split()) > max_words:
      max_words = len(df.iloc[i, 1].split())
  return max_words
