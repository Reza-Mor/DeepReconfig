import json
input_file = "datasets/expert_dbCRW_AND_entry_typeSequence.json"
with open(input_file) as f:
    datas = json.load(f)
    print(len(datas))
    f.close()
