import json
import csv
import os

INPUT = './input_word/'
OUTPUT = './justforatest/'

f = open(OUTPUT + 'input_word.csv', 'a+', encoding='utf-8', newline='')
f_weight = open(OUTPUT + 'input_weight.csv', 'a+', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
csv_writer2 = csv.writer(f_weight)

number_folder_list = os.listdir(INPUT)
print(number_folder_list)
cnt = 0

for number_folder in number_folder_list:
    sub_input = INPUT + number_folder + '/'


    pair_list = os.listdir(sub_input)
    for pair_folder_name in pair_list:
        pair_input = sub_input + pair_folder_name + '/'
        
        if len(os.listdir(pair_input)) == 0:
            continue

        with open(pair_input + '1.json', 'r') as load_f1:
            load_dict1 = json.load(load_f1)

            with open(pair_input + '2.json', 'r') as load_f2:
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
                    load_dict2['OctalInteger'],
                    '0'
                    ])

                load_f2.close()
            
            load_f1.close()


        with open(pair_input + '1_weight.json', 'r') as load_f1:
            load_dict1 = json.load(load_f1)

            with open(pair_input + '2_weight.json', 'r') as load_f2:
                load_dict2 = json.load(load_f2)

                csv_writer2.writerow([
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

        cnt = cnt + 1
        if cnt % 1000 ==0:
            print(cnt)

print(cnt)

