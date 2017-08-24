import matplotlib.pyplot as plt
import json
import numpy as np
result_file = '/home/leo/GitHub/CWS_LSTM/result/model4.txt'
f = open(result_file, 'r')
result2 = json.load(f)
result = np.array(result2)
plot_train2, = plt.plot(result[:,0], 'r-')
plot_valid2, = plt.plot(result[:,2], 'r-.')
f.close()

result_file = '/home/leo/GitHub/CWS_LSTM/result/model5.txt'
f = open(result_file, 'r')
result0 = json.load(f)
result = np.array(result0[:20])
plot_train0, = plt.plot(result[:,0], 'g-')
plot_valid0, = plt.plot(result[:,2], 'g-.')
f.close()


result_file = '/home/leo/GitHub/CWS_LSTM/result/model6.txt'
f = open(result_file, 'r')
result5 = json.load(f)
result = np.array(result5)
plot_train5, = plt.plot(result[:,0], 'b-')
plot_valid5, = plt.plot(result[:,2], 'b-.')
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.xlim([0,20])
plt.legend((plot_train0, plot_valid0, plot_train2, plot_valid2, plot_train5, plot_valid5),
           ('train, dropout:0', 'valid, dropout:0', 'train, dropout:0.2', 'valid, dropout:0.2', 'train, dropout:0.5', 'valid, dropout:0.5'))
plt.show()
f.close()