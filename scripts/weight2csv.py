import numpy
import json
import csv
import os

INPUT = './input_weight/'
OUTPUT = './final_input/'

f = open(OUTPUT + 'input_weight.csv', 'a+', encoding='utf-8', newline='')
csv_writer = csv.writer(f)

two_file = os.listdir(INPUT)

with open(INPUT + two_file[0],'r') as load_f1:
    load_dict1 = json.load(load_f1)

    with open(INPUT + two_file[1],'r') as load_f2:
        load_dict2 = json.load(load_f2)

        csv_writer.writerow([
            load_dict1['BasicType'],load_dict1['Boolean'],
            load_dict1['DecimalInteger'],load_dict1['Identifier'],
            load_dict1['Keyword'],load_dict1['Modifier'],
            load_dict1['Operator'],load_dict1['Separator'],
            load_dict1['Annotation'],load_dict1['Null'],
            load_dict1['DecimalFloatingPoint'],load_dict1['HexInteger'],
            load_dict1['HexFloatingPoint'],load_dict1['BinaryInteger'],
            load_dict1['OctalInteger'],            
            load_dict2['BasicType'],load_dict2['Boolean'],
            load_dict2['DecimalInteger'],load_dict2['Identifier'],
            load_dict2['Keyword'],load_dict2['Modifier'],
            load_dict2['Operator'],load_dict2['Separator'],
            load_dict2['Annotation'],load_dict2['Null'],
            load_dict2['DecimalFloatingPoint'],load_dict2['HexInteger'],
            load_dict2['HexFloatingPoint'],load_dict2['BinaryInteger'],
            load_dict2['OctalInteger']
            ])

        load_f2.close()
    
    load_f1.close()
