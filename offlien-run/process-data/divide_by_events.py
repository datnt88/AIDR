#=================
#==> Libraries <==
#=================
import csv
import sys
import ntpath
import operator
from os.path import basename

#=====================
#==> I/O filenames <==
#=====================
csvfile=sys.argv[1]
name = basename(csvfile)
head, tail = ntpath.splitext(name)

#============
#==> Main <==
#============
with open(sys.argv[1], 'rU') as f:
	reader = csv.reader(f)
	header = next(reader)[:-1]
	sortedlist = sorted(reader, key=operator.itemgetter(3), reverse=True)

prv_row=' '
for row in sortedlist:
	if row[3]!=prv_row:
		outputfile = open('by-events/%s_%s.csv' % (head, row[3]), 'wb')
		writer = csv.writer(outputfile)
		writer.writerow(header)
		writer.writerow(row[:-1])
		prv_row=row[3]
	else:
		writer.writerow(row[:-1])
