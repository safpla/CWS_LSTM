class SmallConfig4(object):
    max_epochs = 30
    batch_size = 256
    ndims = 100
    nhiddens = 100
    layer_num = 1
    dropout_rate = 0.2
    regularization = 0.2
    max_word_len = 4
    max_sent_len = 32
    max_grad_norm = 5.0
    shuffle_data = True
    learning_rate = 0.001
    lr_decay = 0.95
    feature_half_win = 2
    train_file = '/home/leo/GitHub/CWS_LSTM/data/pku_train'
    test_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test_gold'
    pre_training = '/home/leo/GitHub/CWS_LSTM/w2v/c_vecs_100'
    lookup_table_file = '/home/leo/GitHub/CWS_LSTM/model4/lookup_table.txt'
    new_lookup_table = False
    model_save_path = '/home/leo/GitHub/CWS_LSTM/model4/bi_lstm.cws'
    model_load_path = '/home/leo/GitHub/CWS_LSTM/model4/bi_lstm.cws-18'
    restore_from_checkpoint = False

class SmallConfig5(object):
    max_epochs = 30
    batch_size = 256
    ndims = 100
    nhiddens = 100
    layer_num = 1
    dropout_rate = 0.
    regularization = 0.
    max_word_len = 4
    max_sent_len = 32
    max_grad_norm = 5.0
    shuffle_data = True
    learning_rate = 0.001
    lr_decay = 0.95
    feature_half_win = 2
    train_file = '/home/leo/GitHub/CWS_LSTM/data/pku_train'
    test_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test_gold'
    pre_training = '/home/leo/GitHub/CWS_LSTM/w2v/c_vecs_100'
    lookup_table_file = '/home/leo/GitHub/CWS_LSTM/model5/lookup_table.txt'
    new_lookup_table = True
    model_save_path = '/home/leo/GitHub/CWS_LSTM/model5/bi_lstm.cws'
    model_load_path = '/home/leo/GitHub/CWS_LSTM/model5/bi_lstm.cws-30'
    restore_from_checkpoint = False

class SmallConfig6(object):
    max_epochs = 24
    batch_size = 256
    ndims = 100
    nhiddens = 100
    layer_num = 1
    dropout_rate = 0.5
    regularization = 0.
    max_word_len = 4
    max_sent_len = 32
    max_grad_norm = 5.0
    shuffle_data = True
    learning_rate = 0.001
    lr_decay = 0.95
    feature_half_win = 2
    train_file = '/home/leo/GitHub/CWS_LSTM/data/pku_train'
    test_file = '/home/leo/GitHub/CWS_LSTM/data/pku_test_gold'
    pre_training = '/home/leo/GitHub/CWS_LSTM/w2v/c_vecs_100'
    lookup_table_file = '/home/leo/GitHub/CWS_LSTM/model6/lookup_table.txt'
    new_lookup_table = True
    model_save_path = '/home/leo/GitHub/CWS_LSTM/model6/bi_lstm.cws'
    model_load_path = '/home/leo/GitHub/CWS_LSTM/model6/bi_lstm.cws-24'
    restore_from_checkpoint = False