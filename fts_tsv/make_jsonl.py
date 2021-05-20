# Create input files for summ_data

import os
import json
from tqdm import tqdm

title = '../data/dev_title.txt'
sent = '../data/dev_sent.txt'
img = '../data/img'

data_input = []


with open(title) as f_title:
	lines_title = f_title.readlines()
with open(sent) as f_sent:
	lines_sent = f_sent.readlines()

for i in tqdm(range(2000)):
	data = {}
	data["id"] = i
	data["img"]= f"img/{i}.jpg"
	data["title"] = str(lines_title[i])
	data['sent'] = str(lines_sent[i])
	data["label"] = i%2
	data_input.append(data)
	
with open("../data/dev_data.jsonl", "w") as fp:
    for i in range(2000):
    	json.dump(data_input[i],fp)
    	fp.write('\n')

# print(data_input[0])