import json
import csv
import os

INPUT = './bcb_reduced/'
OUTPUT = './label_0_result/'
f = open('./test.csv', 'r', encoding='utf-8')

csv_reader = csv.reader(f)

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

cnt = 0
for row in csv_reader:
    if cnt == 0:
        cnt = cnt + 1
        continue
    
    file1_path = INPUT + row[2] + '/' + row[11] + '/' + row[9]
    file2_path = INPUT + row[2] + '/' + row[12] + '/' + row[10]
    # print(file1_path)
    # print(file2_path)
    f1 = open(file1_path, 'r', encoding='utf-8')
    f2 = open(file2_path, 'r', encoding='utf-8')

    output_path = OUTPUT + str(cnt) + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    f3 = open(output_path + '1.java', 'w')
    f4 = open(output_path + '2.java', 'w')

    lines = f1.readlines()
    tmpcnt = 1
    for line in lines:
        if tmpcnt < int(row[13]) or tmpcnt > int(row[14]):
            tmpcnt = tmpcnt + 1
            continue
        f3.write(line)
        tmpcnt = tmpcnt + 1

    lines = f2.readlines()
    tmpcnt = 1
    for line in lines:
        if tmpcnt < int(row[15]) or tmpcnt > int(row[16]):
            tmpcnt = tmpcnt + 1
            continue
        f4.write(line)
        tmpcnt = tmpcnt + 1

    cnt = cnt + 1
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    if cnt % 10000 == 0:
        print(cnt)
