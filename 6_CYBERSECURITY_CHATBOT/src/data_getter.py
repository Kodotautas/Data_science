import os
import pandas as pd

cwd = os.getcwd()
# one step back in the directory
cwd = os.path.dirname(cwd)

# read the cybersecurity faq excel file
df = pd.read_excel(cwd + '/data/cybersecurity_faq.xlsx')