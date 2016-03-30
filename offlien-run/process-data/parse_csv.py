import csv
import sys
with open(sys.argv[1], 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
	print row[2]
