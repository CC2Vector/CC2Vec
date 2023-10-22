import json
import csv
import os

INPUT = './bcb_result/'
OUTPUT = './input_word/'


def extend_dict(tmp_dict):
    all_class_list = [
    'Operator','Annotation','Separator','DecimalInteger',
    'DecimalFloatingPoint','HexInteger','HexFloatingPoint',
    'BinaryInteger','OctalInteger','Keyword','BasicType','Modifier',
    'Boolean','Null','Identifier'
    ]
    for tmp in all_class_list:
        if tmp not in tmp_dict.keys():
            tmp_dict[tmp] = 'fuxk'
    
    return tmp_dict
 
number_folder_list = os.listdir(INPUT)
print(number_folder_list)
for number_folder in number_folder_list:
    sub_input = INPUT + number_folder + '/'

    sub_output = OUTPUT + number_folder + '/'
    if not os.path.exists(sub_output):
        os.mkdir(sub_output)

    pair_list = os.listdir(sub_input)
    for pair_folder_name in pair_list:
        pair_input = sub_input + pair_folder_name + '/'

        pair_output = sub_output + pair_folder_name + '/'
        if not os.path.exists(pair_output):
            os.mkdir(pair_output)

        two_file = os.listdir(pair_input)
        if len(two_file) != 2:
            continue

        with open(pair_input + two_file[0],'r') as load_f1:
            load_dict1 = json.load(load_f1)

            with open(pair_input + two_file[1],'r') as load_f2:
                load_dict2 = json.load(load_f2)
                
                dict1 = dict()
                for key, value in load_dict1.items():
                    tmpstr1 = str()
                    for key2, value2 in value.items():
                        if ',' in key2:
                            continue
                        tmpstr1 = tmpstr1 + str(key2) +' '
                    dict1[key] = tmpstr1[:-1]

                dict2 = dict()
                for key, value in load_dict2.items():
                    tmpstr2 = str()
                    for key2, value2 in value.items():
                        if ',' in key2:
                            continue
                        tmpstr2 = tmpstr2 + str(key2) +' '
                    dict2[key] = tmpstr2[:-1]
                
                dict1 = extend_dict(dict1)
                dict2 = extend_dict(dict2)



                json_str = json.dumps(dict1, indent=4)
                with open(pair_output + '1.json', 'w') as json_file:
                    json_file.write(json_str)
                    json_file.close()

                json_str = json.dumps(dict2, indent=4)
                with open(pair_output + '2.json', 'w') as json_file:
                    json_file.write(json_str)
                    json_file.close()

                load_f2.close()
            
            load_f1.close()

        

