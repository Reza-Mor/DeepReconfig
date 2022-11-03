import os
import subprocess

def generate_datasets(input_file, max_string_lens, max_graph_sizes, dataset_size):
    assert len(max_graph_sizes) == len(max_string_lens)
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    for i in range(len(max_string_lens)):
        string_len = max_string_lens[i]
        max_graph_size = max_graph_sizes[i]
        #output_file = input_file.split('.')[0] + "_{}by{}".format(max_graph_size, max_graph_size)
        output_file = "datasets/dataset1_{}by{}".format(max_graph_size, max_graph_size)
        command= """python create_dataset.py \
        --input_file {} --dataset_size {} \
        --max_string_len {} --max_graph_size {} \
        --output_file {}""".format(input_file, dataset_size, string_len, max_graph_size, output_file)
        subprocess.run(command, shell=True)

def run_basline(dataset_files, write_k = True):
    for file in dataset_files:
        command = """python run_baseline.py --dataset {} --write_k {}""".format(file, write_k)
        subprocess.run(command, shell=True)

if __name__ == "__main__":

    # generate some datasets!

    # The input file must be of json format outlined in https://rnacentral.org/ (download any of the datasets in json format)
    input_file = 'datasets/expert_dbCRW_AND_entry_typeSequence.json'
    # ideally we would want the string len to be large enough so the
    # number of bounds formed is larger than the max_graph_size
    max_string_lens = [250] #[80, 150, 200, 350, 380]
    # max_graph_sizes denotes the number of nodes on one side of a bipirtite graph
    max_graph_sizes = [20] #[10, 25, 50, 85, 100]
    dataset_size = 800

    dataset_files = ['datasets/dataset1_20by20']
    # COMMENT OUT IF DON'T WANT TO MAKE DATASETS
    generate_datasets(input_file, max_string_lens, max_graph_sizes, dataset_size)

    run_basline(dataset_files, write_k = True)


