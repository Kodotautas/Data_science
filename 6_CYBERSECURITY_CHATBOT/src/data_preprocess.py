import json
import os
import pandas as pd

def df_to_tuple(df):
    """Convert a dataframe to a list of tuples.
    Args:
        df (pandas.DataFrame): The dataframe to convert.
    Returns:
        list: A list of tuples.
    """
    # convert the DataFrame to a list of records (tuples)
    records = df.to_records(index=False)

    # extract the questions and answers from the tuples and create a list of tuples
    tuples = [(record[0], record[1]) for record in records]

    return tuples

def convert_to_jsonl(data):
    # Write the data to a JSONL file
    with open(os.getcwd() + "/data/security_faq.jsonl", "w") as f:
        for input_text, output_text in data:
            example = {"input": input_text, "output": output_text}
            f.write(json.dumps(example))
            f.write("\n")