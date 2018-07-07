import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_in')
parser.add_argument('--dataset_out')
parser.add_argument('--delimiter_in')
parser.add_argument('--delimiter_out')
args = vars(parser.parse_args())

dataset_in_path = args['dataset_in']
dataset_out_path = args['dataset_out']
delimiter_in = args['delimiter_in']
delimiter_out = args['delimiter_out']

with open(dataset_in_path, 'r') as input:
    with open(dataset_out_path, "w") as output:
        for row in input.readlines():
            splits = row.split(delimiter_in)
            output.write(splits[0] + delimiter_out + splits[1] + delimiter_out + splits[2])

input.close()
output.close()