import src.test as test
import sys

"""
# example
sentence = u'人类社会前进的航船就要驶入0世纪的新航程。中国人民进入了向现代化建设第三步战略目标迈进的新征程。在这个激动人心的时刻,我很高兴通过中国国际广播电台、中央人民广播电台和中央电视台,向全国各族人民,向香港特别行政区同胞、澳门特别行政区同胞和台湾同胞、海外侨胞,向世界各国的朋友们,致以新世纪第一个新年的祝贺!'
result = word_seg(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print(rss)
"""
print_perf = False
if len(sys.argv) > 3:
    gold_file = sys.argv[3]
    print_perf = True
else:
    gold_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test_gold'

if len(sys.argv) > 2:
    output_file = sys.argv[2]
else:
    output_file = '/home/leo/GitHub/CWS_LSTM/data/pku_result_model6'

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test'
    print_perf = True


test.process(input_file, output_file)
if print_perf:
    Precision, Recall, F1_measure = test.performance(gold_file, output_file)
    print(Precision, Recall, F1_measure)