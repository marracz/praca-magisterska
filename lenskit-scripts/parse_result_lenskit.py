#! python
import re
import sys

if len(sys.argv) < 2:
	print('you need to pass file name to parse as parameter')
	exit(1)

content = list([line for line in open(sys.argv[1]).read().split('\n') if len(line) > 0])

#print(content)

for row in content:
	splits = row.split(",")
	print(splits)
	for s in splits:
		val = s[s.find("[")+1:s.find("]")]
		print(val)