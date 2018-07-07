#! python
import re
import sys

if len(sys.argv) < 2:
	print('you need to pass file name to parse as parameter')
	exit(1)

content = list([line for line in open(sys.argv[1]).read().split('\n') if len(line) > 0])
content = list([line for line in content if not line.startswith('log4j')])
chunk_size = 4
if len(content) % chunk_size is not 0:
	print('WARNING - file contains {} lines which is indivisible by chunk_size={}'.format(len(content), chunk_size))

rows = list(zip(*[iter(content)]*chunk_size))

print('-;-;-;-;jar;main class;task;dataset;model class;algorithm class;neighbourhood size;relevant threshold;start time;-;f1;precision;recall;ndcg;end time')
for row in rows:
	for java_arg in row[0][1:-1].split(','):
		print(java_arg.strip(), end=';')
	print(row[1], end=';')
	for metric in row[2].split(' '):
		print(re.sub(r'.*=', '', metric.replace(',', '').replace('.', ',')), end=';')
	print(row[3], end=';')
	print()
		