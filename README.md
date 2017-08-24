# CWS_LSTM
use bi-LSTM to do Chinese Word Segmentation

The executable script is CWS_LSTM/script.py
Before you can run script.py, you need to change the absolute path in CWS_LSTM/src/config.py to your settings.
Then you can run script.py:
  $python script.py input_file_name, output_file_name, gold_file_name

Input:
    input_file_name: The absolute path of the input file which was going to be processed
    output_file_name: The absolute path of the output file name
    gold_file_name: The ground-truth segmentation of the input file. It is used to evaluate the performance of the result. It is an optional  parameter. If it is given, the script will print the precision, recall and f1-measure.

