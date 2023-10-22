# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import json
import os
import io

from javalang import tokenizer


file_path = '../bcb_reduced/bcb_reduced/'
save_file = './result.txt'


def one_file(raw_file):

    file = io.open(raw_file, 'r', encoding='utf-8')
    try:
        tokens = list(tokenizer.tokenize(file.read()))
    except:
        with open('./failed.txt', 'a+') as failed_file:
            failed_file.write(raw_file + '\n')
            failed_file.close()
            return

    with io.open(save_file, 'a+', encoding='utf-8') as f:
        pre = 1
        for line in tokens:
            if line.position.line != pre:
                f.write('\n'.decode('utf-8'))
                pre = line.position.line
            f.write(line.value.decode('utf-8'))
            f.write(' '.decode('utf-8'))
        
        f.write('\n\n'.decode('utf-8'))
        f.close()

    return

folders = os.listdir(file_path)
for folder in folders:
    _path = file_path + folder + '/'
    sub_folders = os.listdir(_path)
    for sub_folder in sub_folders:
        path = _path + sub_folder + '/'
        
        files = os.listdir(path)
        for file_name in files:
            final_path = path + file_name
            print(final_path)
            one_file(final_path)
