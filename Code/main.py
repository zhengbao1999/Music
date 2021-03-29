# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from __future__ import print_function

import numpy as np
from scipy.integrate import simps
from numpy import trapz
def shiftLeft(float_arr):
    float_arr = float_arr.insert(0, 0.0)
    return float_arr


def shiftRight(float_arr):
    float_arr = float_arr.append(0.0)
    return float_arr


def shiftUp(float_arr):
    for arr_element in float_arr:
        arr_element += 0.01
    return float_arr


def shiftDown(float_arr):
    for arr_element in float_arr:
        arr_element -= 0.01
    return float_arr

file1 = "/Users/ejingtong/PycharmProjects/pythonProject1/venv/genre_500_pop20_melody/1.txt"
with open(file1) as f:
    content = f.readlines()


file2 = "/Users/ejingtong/PycharmProjects/pythonProject1/venv/genre_500_pop20_melody/2.txt"
with open(file2) as f2:
    content2 = f2.readlines()

pitches = []
pitches2 = []
# f = open(r'D:\UNC\2021 Spring\Musical Similarity\Lakh\genre_500_pop20_melody\1.txt', "r")
for line in content:
    # print(line)
    pitches.append(line.strip('\n').split("\t")[1])

for line in content2:
    pitches2.append(line.strip('\n').split("\t")[1])

float_pitch = []
float_pitch2 = []
float_pitch3 = []
for x in pitches:
    if x != '0.0':
        float_pitch.append(float(x))

for x in pitches2:
    if x != '0.0':
        float_pitch2.append(float(x))

original_float_pitch = float_pitch
original_float_pitch2 = float_pitch2

size1 = len(float_pitch)
size2 = len(float_pitch2)
if size2 > size1:
    min_length = size1
    max_length = size2
    longer = 2
else:
    max_length = size1
    min_length = size2
    longer = 1

min_pitch = min(min(float_pitch), min(float_pitch2))
max_pitch = max(max(float_pitch), max(float_pitch2))


a = 0
b = 0.0
total_area = 9999999
i = 0
while b < max_pitch - min_pitch:
    a = 0
    while a < min_length:
        size1 = len(float_pitch)
        size2 = len(float_pitch2)
        if size2 > size1:
            min_length = size1
            max_length = size2
            longer = 2
        else:
            max_length = size1
            min_length = size2
            longer = 1
        area = 0
        while i < min_length:
            area += (abs(float_pitch[i] - float_pitch2[i])) * 0.01
            i = i + 1
        while i < max_length:
            if longer == 1:
                area += abs(float_pitch[i]) * 0.01
            else:
                area += abs(float_pitch2[i]) * 0.01
            i = i + 1
        if area < total_area:
            total_area = area
        float_pitch.insert(0, 0.0)
        a = a + 1
        i = 0
    b = b + 1
    float_pitch = original_float_pitch
    for abc in float_pitch:
        abc = abc + 0.01
print(total_area)

