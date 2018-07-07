import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_in')
parser.add_argument('--dataset_out_prefix')
args = vars(parser.parse_args())

dataset_in = args['dataset_in']
dataset_out_prefix = args['dataset_out_prefix']

folds = []

for x in range(10):
    folds.append([])
    #print(folds)

header = ""
with open(dataset_in, 'r') as input:
    all_lines = input.readlines()
    header = all_lines[0]
    for row in all_lines[1:]:
	fold_num = random.randint(0, 9)
	folds[fold_num].append(row)	
	
input.close()

for x in range(10):
    test_dataset_out = dataset_out_prefix + str(x) + '_test.csv'
    training_dataset_out = dataset_out_prefix + str(x) + '_training.csv'
    with open(test_dataset_out, 'w') as test_output:
	test_output.write(header)
	test_output.writelines(folds[x])
    with open(training_dataset_out, 'w') as training_output:
	training_output.write(header)
	for y in range(10):
	    if x != y:
                training_output.writelines(folds[y])
