import json
import shelve

input_file = "datasets/expert_dbCRW_AND_entry_typeSequence.json"
with open(input_file) as f:
    datas = json.load(f)
    print(len(datas))
    f.close()

dataset_path = 'datasets/dataset1_20by20'
db = shelve.open(dataset_path)
dataset_size, max_graph_size, max_string_length =  db['non_zero_k'], db['max_graph_size'], db['max_string_length']
max_reward =  max_graph_size * 2
db.close()
    
print("Training a model on a dataset of size {} of {}by{} graphs (RNA string length: {})".format(dataset_size, max_graph_size, max_graph_size, max_string_length))