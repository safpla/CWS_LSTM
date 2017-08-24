import src.config
import src.tools
import re
import time
import json
import numpy as np
import tensorflow as tf
import src.tools as tools
from sklearn.model_selection import train_test_split

def test_epoch(dataset):
    """Testing"""
    _costs = 0.0
    _accs = 0.0
    _batch_size = 256
    data_size = dataset.y.shape[0]
    batch_num = int(data_size / _batch_size)
    fetches = [accuracy, cost, y_pred]
    for i in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs: X_batch,
                     y_inputs: y_batch,
                     lr: 1e-5,
                     batch_size: _batch_size,
                     keep_prob: 1.0}
        _acc, _cost, _y_pred = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
        # print(_y_pred)
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

def text2ids(text):
    feature_half_win = config.feature_half_win
    words = list(text)
    ids = []
    for i in range(feature_half_win):
        ids.append(character_idx_map['start'])

    ids.extend(list(character_idx_map[word] if word in character_idx_map else 0 for word in words))

    for i in range(feature_half_win):
        ids.append(character_idx_map['stop'])
    _data = np.zeros([1, max_sent_len, feature_half_win * 2 + 1], np.int32)

    for i in range(min(len(text), max_sent_len)):
        _data[0, i, :] = ids[i:i+feature_half_win * 2 + 1]

    return _data

def viterbi(y_pred, A):
    value = np.zeros([max_sent_len, 4])
    previous = -1 * np.ones([max_sent_len, 4], dtype=np.int32)
    value[0,0] = y_pred[0]['B']
    value[0,3] = y_pred[0]['S']
    num2tag =['B', 'M', 'E', 'S']
    for layer in range(1, len(y_pred)):
        for node in range(4):
            for pre_node in range(4):
                if ((A[pre_node, node] != 0) and
                    (value[layer-1, pre_node] + A[pre_node, node] + y_pred[layer][num2tag[node]]) > value[layer,node]):
                    value[layer, node] = value[layer-1, pre_node] + A[pre_node, node] + y_pred[layer][num2tag[node]]
                    previous[layer, node] = pre_node
    longest = 0
    for node in range(4):
        if value[len(y_pred)-1,node] > longest:
            longest = value[len(y_pred)-1,node]
            node_now = node

    path_rev = []
    for layer in range(len(y_pred)-1,-1,-1):
        path_rev.append(num2tag[node_now])
        node_now = previous[layer, node_now]

    path_rev.reverse()
    return path_rev

def simple_cut(text):
    if text:
        words = []
        while True:
            text_len = min(len(text), max_sent_len)
            X_batch = text2ids(text[:text_len])
            fetches = [y_pred]
            feed_dict = {X_inputs:X_batch, lr:1.0, batch_size:1, keep_prob:1.0}
            _y_pred = sess.run(fetches, feed_dict)[0][:text_len]
            nodes = [dict(zip(['B','M','E','S'], each[1:])) for each in _y_pred]
            tags = viterbi(nodes, A)

            for i in range(len(tags)):
                if tags[i] in ['S','B']:
                    words.append(text[i])
                else:
                    words[-1] += text[i]
            if len(text) <= max_sent_len:
                break
            else:
                text = text[max_sent_len:]

        return words
    else:
        return []

def word_seg(sentence):
    not_cuts = re.compile(u'([a-zA-Z ]+)|[\W]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:]))
    return result

def process(input_file, output_file):
    fin = open(input_file, 'r')
    fout = open(output_file, 'w')
    for lines in fin.readlines():
        result = word_seg(lines.strip())
        len_result = len(result)
        rss = ''
        for index, each in enumerate(result):
            if (each in ['…', '—']) and (result[(index+1) % len_result] in ['…', '—']):
                rss = rss + each
            else:
                rss = rss + each + '  '
        if lines[-1] == '\n':
            rss = rss[:-2] + '\n'
        fout.write(rss)
    fin.close()
    fout.close()

def performance(gold_file, result_file):
    fgold = open(gold_file, 'r')
    fresult = open(result_file, 'r')
    tp, fp, tn, fn = 0, 0, 0, 0
    gold = fgold.readline()
    result = fresult.readline()
    while (gold != '') and (result != ''):
        point1 = 0
        point2 = 0
        while (point1 < len(gold)) and (point2 < len(result)):
            space_found1 = False
            space_found2 = False
            while gold[point1] == ' ':
                space_found1 = True
                point1 += 1
            while result[point2] == ' ':
                space_found2 = True
                point2 += 1
            if space_found1 and space_found2:
                tp += 1
            if space_found1 and (not space_found2):
                fn += 1
            if (not space_found1) and space_found2:
                fp += 1
            if (not space_found1) and (not space_found2):
                tn += 1
            point1 += 1
            point2 += 1
        gold = fgold.readline()
        result = fresult.readline()
    p = tp/(tp+fp+1e-8)
    r = tp/(tp+fn+1e-8)
    f1 = 2*p*r/(p+r+1e-8)
    fgold.close()
    fresult.close()
    return p, r, f1

# load parameters
config = src.config.SmallConfig6()
ndims = config.ndims
train_file = config.train_file
test_file = config.test_file
pre_training = config.pre_training
lookup_table_file = config.lookup_table_file
max_sent_len = config.max_sent_len
f = open(lookup_table_file)
lookup = json.load(f)
f.close()
Cemb = np.array(lookup['Cemb'])
character_idx_map = lookup['character_idx_map']

# prepare data
data, label = tools.prepareData(train_file, character_idx_map, config)
#   data: sentences in the form of character index
#         shape:[num_sent, max_sent_len, feature_win]
#         dtype:int32
#   label: sentences in the form of tags
#         shape:[num_sent, max_sent_len]
shuffle_data = True
X_train, X_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, random_state=1)
data_train = tools.BatchGenerator(X_train, y_train, shuffle=shuffle_data)
data_valid = tools.BatchGenerator(X_valid, y_valid, shuffle=shuffle_data)

X_test, y_test = tools.prepareData(test_file, character_idx_map, config)
data_test = tools.BatchGenerator(X_test, y_test, shuffle=shuffle_data)

# load model
model_path = config.model_load_path
meta_path = model_path + '.meta'

sess = tf.Session()
saver = tf.train.import_meta_graph(meta_path)
saver.restore(sess, model_path)
graph = tf.get_default_graph()
X_inputs = graph.get_operation_by_name('X_inputs').outputs[0]
y_inputs = graph.get_operation_by_name('y_inpupts').outputs[0]
lr = graph.get_operation_by_name('lr').outputs[0]
batch_size = graph.get_operation_by_name('batch_size').outputs[0]
keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
accuracy = tf.get_collection('accuracy')[0]
cost = tf.get_collection('cost')[0]
y_pred = tf.get_collection('y_pred')[0]

# calculate transition matrix A
A = np.zeros([4, 4], int)
for sent, tags in zip(data, label):
    ind = 1
    while sent[ind][2] != 0:
        A[tags[ind-1]-1, tags[ind]-1] += 1
        ind += 1
A = (A.T / np.sum(A,1)).T
