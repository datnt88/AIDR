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
	sortedlist = sorted(reader, key=operator.itemgetter(2), reverse=True)

prv_row=' '
for row in sortedlist:
	if row[2]!=prv_row:
		outputfile = open('by-classes/%s_%s.csv' % (head, row[2]), 'wb')
		writer = csv.writer(outputfile)
		#writer.writerow(header)
		writer.writerow(row[:])
		prv_row=row[2]
	else:
		writer.writerow(row[:])
