import re

def remove_space(input_file, output_file):
    fin = open(input_file, 'r')
    fout = open(output_file, 'w')
    for line in fin.readlines():
        fout.write(re.sub(' ', '', line))

    fin.close()
    fout.close()

input_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test_gold'
output_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test'
remove_space(input_file, output_file)