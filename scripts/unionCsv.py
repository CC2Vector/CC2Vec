import os
import csv

f1 = open('./label1/input_word.csv', 'r')
f2 = open('./label0/input_word.csv', 'r')

f = open('./final_input/input_word.csv', 'w')

lines = f1.readlines()

cnt1 = 0
for line in lines:
    f.write(line)
    cnt1 = cnt1 + 1
    if(cnt1 % 10000 == 0):
        print(cnt1)

lines = f2.readlines()

flagg = 0
cnt2 = 0
for line in lines:
    if flagg == 0:
        flagg = 1
        continue
    f.write(line)
    cnt2 = cnt2 + 1
    if(cnt2 % 10000 == 0):
        print(cnt2)

f1.close()
f2.close()
f.close()

print(cnt1)
print(cnt2)
print(cnt1 + cnt2)