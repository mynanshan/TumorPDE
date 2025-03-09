import pandas as pd

patient="XU-XIAO"

params_records = pd.read_csv("results/parameters.txt", sep="\t")
print(params_records)
first_record = params_records[params_records['Patient'] == patient].iloc[0]
print(first_record)