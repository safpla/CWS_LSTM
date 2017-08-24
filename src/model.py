import src.tools as tools
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
import numpy as np
import src.config
import time
import json


# define bi_list
def bi_lstm(X_input, keep_prob, batch_size, _weights, config):
    max_sent_len = config.max_sent_len
    ndims = config.ndims
    feature_half_win = config.feature_half_win
    nchar = feature_half_win * 2 + 1

    nhiddens = config.nhiddens
    layer_num = config.layer_num
    # lookup table
    inputs = tf.nn.embedding_lookup(_weights['Cemb'], X_input)
    # concatenate# modify the input shape
    inputs = tf.reshape(inputs, [batch_size, max_sent_len, nchar * ndims])

    # LSTM layer
    lstm_fw_cell = rnn.BasicLSTMCell(nhiddens, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = rnn.BasicLSTMCell(nhiddens, forget_bias=1.0, state_is_tuple=True)
    # dropout
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # multi-layer LSTM
    cell_fw = rnn.MultiRNNCell([lstm_fw_cell] * layer_num, state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_bw_cell] * layer_num, state_is_tuple=True)
    # init
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
    # bi_lstm compute
    inputs = tf.unstack(inputs, max_sent_len, 1)  # modify the input shape
    outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw,
                                                 cell_bw,
                                                 inputs,
                                                 initial_state_fw=initial_state_fw,
                                                 initial_state_bw=initial_state_bw,
                                                 dtype=tf.float32
                                                 )
    output = tf.reshape(tf.concat(outputs, 1), [-1, ndims * 2])
    logits = tf.add(tf.matmul(output, _weights['softmax_w']), _weights['softmax_b'], name='y_pred')
    return logits


def train_model(config):

    def test_epoch(dataset):
        """Testing"""
        _costs = 0.0
        _accs = 0.0
        _batch_size = train_batch_size
        data_size = dataset.y.shape[0]
        batch_num = int(data_size / _batch_size)
        fetches = [accuracy, cost]
        for i in range(batch_num):
            X_batch, y_batch = dataset.next_batch(_batch_size)
            feed_dict = {X_inputs: X_batch,
                         y_inputs: y_batch,
                         lr: 1e-5,
                         batch_size: train_batch_size,
                         keep_prob: 1.0}
            _acc, _cost = sess.run(fetches, feed_dict)
            _accs += _acc
            _costs += _cost
        mean_acc = _accs / batch_num
        mean_cost = _costs / batch_num
        return mean_acc, mean_cost

    # load parameters
    max_epochs = config.max_epochs
    train_batch_size = config.batch_size
    ndims = config.ndims
    max_sent_len = config.max_sent_len
    train_file = config.train_file
    feature_half_win = config.feature_half_win
    pre_training = config.pre_training
    nchar = feature_half_win * 2 + 1
    max_grad_norm = config.max_grad_norm
    lr_decay = config.lr_decay
    shuffle_data = config.shuffle_data
    dropout_rate = config.dropout_rate
    lookup_table_file = config.lookup_table_file
    new_lookup_table = config.new_lookup_table
    model_save_path = config.model_save_path
    model_load_path = config.model_load_path
    restore_from_checkpoint = config.restore_from_checkpoint

    print('Building model')
    if new_lookup_table:
        # initialize embedding lookup table
        Cemb, character_idx_map = tools.initCemb(ndims, train_file, pre_training)
        lookup = {'Cemb': Cemb.tolist(), 'character_idx_map': character_idx_map}
        f = open(lookup_table_file,'w')
        json.dump(lookup,f)
        f.close()
    else:
        # load previous lookup table
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
    X_train, X_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, random_state=1)
    data_train = tools.BatchGenerator(X_train, y_train, shuffle=shuffle_data)
    data_valid = tools.BatchGenerator(X_valid, y_valid, shuffle=shuffle_data)

    # define parameters
    weights = {
        'Cemb': tf.Variable(Cemb, dtype=tf.float32, name='Cemb'),
        'softmax_w': tf.Variable(tf.truncated_normal([ndims*2, 5], stddev=0.1), name='softmax_W'),
        'softmax_b': tf.Variable(tf.constant(0.1, shape=[5]), name='softmax_b')
    }

    # build model
    lr = tf.placeholder(tf.float32, name='lr')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    X_inputs = tf.placeholder(tf.int32, [None, max_sent_len, nchar], name='X_inputs')
    y_inputs = tf.placeholder(tf.int32, [None, max_sent_len], name='y_inpupts')

    y_pred = bi_lstm(X_inputs, keep_prob, batch_size, weights, config)
    correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32),
                                  tf.reshape(y_inputs, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]),
                                                                         logits=y_pred))
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=tf.contrib.framework.get_or_create_global_step())
    print('Finish creating the bi-list model.')

    print('Start training')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('cost', cost)

    batch_num = int(data_train.y.shape[0] / train_batch_size)
    if restore_from_checkpoint:
        saver.restore(sess, model_load_path)
    result = []
    for epoch in range(max_epochs):
        _lr = 1e-3
        if epoch > 5:
            _lr = _lr * (lr_decay ** (epoch - 5))
        print('EPOCH %d, lr=%g' % (epoch+1, _lr))

        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        show_accs = 0.0
        show_costs = 0.0
        display_epoch_num = 200000
        for batch in range(batch_num):
            fetches = [accuracy, cost, train_op]
            X_batch, y_batch = data_train.next_batch(train_batch_size)
            feed_dict = {X_inputs: X_batch,
                         y_inputs: y_batch,
                         lr: _lr,
                         batch_size: train_batch_size,
                         keep_prob: 1 - dropout_rate}
            _acc, _cost, _ = sess.run(fetches, feed_dict)
            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost

            if (batch + 1) % display_epoch_num == 0:
                valid_acc, valid_cost = test_epoch(data_valid)
                print('\ttraining acc=%g, cost=%g; valid acc=%g, cost=%g'
                      % (show_accs / display_epoch_num,
                         show_costs / display_epoch_num,
                         valid_acc,
                         valid_cost)
                      )
                show_accs = 0.0
                show_costs = 0.0

        mean_acc = _accs / batch_num
        mean_cost = _costs / batch_num
        # save_path = saver.save(sess, model_save_path, global_step=31)
        # print('the save path is ', save_path)

        if (epoch + 1) % 3 == 0: # save the model by every 3 epoches
            save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
            print('the save path is ', save_path)

        valid_acc, valid_cost = test_epoch(data_valid)
        print('\ttrain: acc=%g, cost=%g, valid: acc=%g, cost=%g, speed=%g s/epoch'
              % (mean_acc, mean_cost, valid_acc, valid_cost, time.time()-start_time))
        result.append([mean_acc, mean_cost, valid_acc, valid_cost])

    f = open('/home/leo/GitHub/CWS_LSTM/result/model6.txt', 'w')
    json.dump(result, f)
    f.close()
    return 0


if __name__ == '__main__':
    config = src.config.SmallConfig6()
    train_model(config)
