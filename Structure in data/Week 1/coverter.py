# -*- coding: utf-8 -*-
import csv

# with open('/Users/oneinchman/Documents/Git-repositories/MIPT-ML/Structure in data/Week 1/checkins.dat', 'r') as file:
# 	with open('/Users/oneinchman/Documents/Git-repositories/MIPT-ML/Structure in data/Week 1/checkins.csv', 'w') as nfile:
# 		writer = csv.writer(nfile)
# 		for line in file.readlines():
# 			new_line = ''
# 			tmp_list = list(map(str.strip, line.split('|')))
# 			for idx, elem in enumerate(tmp_list):
# 				new_line += elem + ',' if idx < len(tmp_list) else ''

# 			writer.writerow(new_line)
			# print(new_line)

with open('/Users/oneinchman/Documents/Git-repositories/MIPT-ML/Structure in data/Week 1/checkins.dat', 'r') as fin:
	with open('/Users/oneinchman/Documents/Git-repositories/MIPT-ML/Structure in data/Week 1/checkins.csv', 'w') as fout:
	    for line in fin:
	        newline = list(map(str.strip, line.split('|')))
	        if len(newline) == 6 and newline[3] and newline[4]:
	            csv.writer(fout).writerow(newline)

		