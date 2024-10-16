import pandas as pd

DROPPING_ARTIFACTS = True

def freq_amp_pair_string_to_list(string):
    string = string[1:-1]
    string = string.split(", ")
    return [[float(string[i][1:]), float(string[i+1][:-1])] for i in range(0, len(string), 2)]

def post_pad_list(lst, length):
    return lst + [0] * (length - len(lst))

# def align(short_list, long_list):
#     if len(short_list) > len(long_list):
#         raise ValueError("Short list is longer than long list")
#     else:
#         return post_pad_list(short_list, len(long_list))


processed_dataset = pd.read_csv("extracted/archive/DHD/processed_dataset_mini.csv")
if DROPPING_ARTIFACTS:
    processed_dataset = processed_dataset[processed_dataset["label_value"] != 3]

processed_dataset["extracted_freq"] = processed_dataset["extracted_freq"].apply(freq_amp_pair_string_to_list)
processed_dataset["freq_len"] = processed_dataset["extracted_freq"].apply(len)
processed_dataset = processed_dataset.sort_values(by="freq_len", ascending=False).reset_index(drop=False)

max_freq_len = processed_dataset["freq_len"].max()
processed_dataset["extracted_freq"] = processed_dataset["extracted_freq"].apply(post_pad_list, length=max_freq_len)

# Implementing the rnn
# classification as: {"normal" : 0, "extrahls" : 1, "murmur" : 2, "artifact" : 3, "extrastole" : 4}
# since we are dropping artifacts, we will have 4 classes with corresponding processed_dataset["label_value"]: 0, 1, 2, 4
################## CODE TO FIX IS BELOW, DO NOT CHANGE CODE ABOVE ##################
