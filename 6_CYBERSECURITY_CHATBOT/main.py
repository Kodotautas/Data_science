import os
import random
import pandas as pd
from src.data_preprocess import df_to_tuple, convert_to_jsonl
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()


# ------------------------------ DATA PREPROCESSING ------------------------------ #
# read the cybersecurity faq excel file
df = pd.read_excel(cwd + '/data/security_faq.xlsx')

# convert the dataframe to a list of tuples
data = df_to_tuple(df)
convert_to_jsonl(data)