import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_in')
parser.add_argument('--dataset_out')
args = vars(parser.parse_args())

dataset_in_path = args['dataset_in']
dataset_out_path = args['dataset_out']

with open(dataset_in_path, 'r') as input:
    with open(dataset_out_path, "w") as output:
        for row in input.readlines()[1:]:
            output.write(row)

input.close()
output.close()