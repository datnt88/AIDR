#=================
#==> Libraries <==
#=================
import csv
import sys
from os.path import basename
import ntpath
import os

#=====================
#==> I/O filenames <==
#=====================
csvfile=sys.argv[1]
name = basename(csvfile)
head, tail = ntpath.splitext(name)

outputfile = open('%s_relabeled.csv' % head, 'wb')
writer = csv.writer(outputfile)

#============
#==> Main <==
#============
a=b=c=0
with open(csvfile, 'rU') as f:
	reader = csv.reader(f)
	writer.writerow(next(reader))#[:-1])
	for row in reader:
		if row[2] == "Not related or irrelevant":
			row[2]="not informative"
		else:
			row[2]="informative"
		writer.writerow(row)
