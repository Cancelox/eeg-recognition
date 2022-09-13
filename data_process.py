import copy
from math import pow, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import LearningRateScheduler

from models import Shallow_Fusion1
from models import Shallow_Fusion_LSTM

def bandpass(data, f1, f2, step, sfreq):
    wn1 = 2*f1/sfreq
    wn2 = 2*f2/sfreq
    #redata = data #copy.deepcopy(data)
    #带通滤波
    for subject in data:
        for session in data[subject]:
            for channel in data[subject][session]:
                for event in range(len(data[subject][session][channel])):
                    #带通滤波
                    [b, a] = signal.butter(step, [wn1, wn2], 'bandpass')  #8阶
                    data_bandpass = signal.filtfilt(b, a, data[subject][session][channel][event])
                    data[subject][session][channel][event] = data_bandpass
                    #print("length of one event is:", len(data[subject][session][channel][event]))
    return data


#滑窗法进行数据增强+噪声混合数据
#将加噪声数据加入原始数据形成新数据集
def preprocess_signal(ori_data, start_time, slide_len, segment_len, num, sfreq):
    mu = 0
    sigma = 0.2
    processed_data = list()

    #先添加原始数据
    left = int((start_time + 0 * slide_len) * sfreq)
    right = int(left + sfreq * segment_len)  # segment_len = 2s
    data_original = ori_data[left:right] # / max(abs(ori_data))
    # data_original = data_original - np.mean(data_original)
    processed_data.append(data_original)

    '''
    #再添加含噪声数据：可用84%以上准确率
    noise_mod_val = 2
    for i in range(num):
        #left = int((start_time + 0 * slide_len)*sfreq)
        #right = int(left + sfreq*segment_len)  #segment_len = 2s
        data_foo = ori_data[left:right] #/ max(abs(ori_data))
        stddev_t = np.std(data_foo)
        rand_t = np.random.rand(len(data_foo))
        rand_t = rand_t - 0.5
        to_add_t = rand_t * stddev_t / noise_mod_val
        data_t = data_foo + to_add_t
        processed_data.append(data_t)
    #'''

    return processed_data

# 数据增强
def get_input_data(data, channel=('EEG:C3', 'EEG:Cz', 'EEG:C4')):
    input_data = list()
    num = 1 # 6
    mu_gauss = 0
    sigma_gauss = 0.2
    num_segments = len(data[channel[0]]) # num_segments=2  EEG:C3
    for idx in range(num_segments):
        linedata = list() # 提取data为list形式， idx 1，channel 1 -> n
        for chn in channel:
            linedata.append(data[chn][idx])
        input_image = np.array(linedata)# 转化为数组，变为图像
        # 0均值标准化/Z-score Normalization:Z归一化0~1之间
        # 极差变换法标准化
        mean = input_image.mean()
        sigma = input_image.std()
        data_normalized = input_image
        data_normalized = (data_normalized - mean) / sigma
        data_normalized = (data_normalized - np.min(data_normalized)) / np.ptp(data_normalized)
        input_data.append(data_normalized)
        # input_data.append(input_image)


    # 时域加噪声
    #for i in range(num):
    data_noise = copy.deepcopy(data_normalized)
    #print(data_noise.shape)
    noise_mod_val = 2
    stddev_t = np.std(data_noise) #标准差
    rand_t = np.random.rand(data_noise.shape[0], data_noise.shape[1])
    rand_t = rand_t - 0.5
    to_add_t = rand_t * stddev_t / noise_mod_val
    data_t = data_noise + to_add_t
    input_data.append(data_t)

    '''
    # 数据增强乘
    data_mult1 = copy.deepcopy(data_normalized)#复制一个新的，并且不改变原有变量
    data_mult2 = copy.deepcopy(data_normalized)
    mult_mod = 0.05
    mul_aug1 = data_mult1*(1+mult_mod)
    input_data.append(mul_aug1)
    mul_aug2 = data_mult2*(1-mult_mod)
    input_data.append(mul_aug2)
    '''

    return input_data


def process(data, labels, sfreq):
    start_time = 3.5  # 从第3s开始
    time_slides = 3.5  # 每次右移0.2s
    window_length = 3.5  # 窗长为2s
    segments_num = 1  # 数据加噪声10倍

    preprocessed_data = dict()
    for subject in data:
        if subject not in preprocessed_data:
            preprocessed_data[subject] = dict()
        for session in data[subject]:
            df_trials_data = pd.DataFrame()
            for channel in data[subject][session]:
                session_data = data[subject][session][channel] #{list:120} {ndarray:(2000,)}
                trials_processed_data = list()
                for trial_data in session_data:
                    '''preprocess_signal'''
                    processed_data = preprocess_signal(trial_data, start_time, time_slides, window_length,
                                                       segments_num, sfreq) #trial_data 是 ori_data
                    trials_processed_data.append(processed_data) #按每个channel {list:120} {list:11} {ndarray:(500,)}
                df_trials_data[channel] = trials_processed_data #{DataFrame:(120,6)}  120：每个里面11个500长的数据
            preprocessed_data[subject][session] = df_trials_data

    for subject in preprocessed_data:
        for session in preprocessed_data[subject]:
            preprocessed_data[subject][session]['input data'] = preprocessed_data[subject][session].apply(get_input_data, axis=1)
            '''get_input_data'''
    print("行索引：", preprocessed_data["subject1"]["session1"].index)
    print("列索引：", preprocessed_data["subject1"]["session1"].columns)
    print(preprocessed_data['subject1']['session1']['input data'])
    return preprocessed_data, labels

##########################################################################################################################################################
#数据转为数组  标签转为0和1 (数据必须是数组形式才能运行cross validation)
def arrange_data(data, labels):
    output_data = list()
    output_labels = list()
    for idx in range(len(data)): # data: ndarray:(648,4,3,875)
        for segment in data[idx]: # segment:ndarray: (3,875) 循环4遍
            output_data.append(np.expand_dims(segment, axis=2)) # ndarray: (3,875,1)
            if labels[idx][0] == 1:
                output_labels.append(0)
            else:
                output_labels.append(1)
    output_data = np.array(output_data) # ndarray:(2592,3,875,1)
    output_labels = np.array(output_labels) # ndarray:(2592,)

    return output_data, output_labels

# 指数衰减学习率
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.75  # 0.6
    epochs_drop = 4  # 5
    lrate = init_lrate*pow(drop, floor(1+epoch))/epochs_drop
    return lrate

#交叉验证
def run_classification(data, labels, session=(1, 2, 3, 4, 5)):
    kf = KFold(n_splits=5, shuffle=False) #n倍交叉验证
    #classification_acc = pd.DataFrame()
    result = list()
    for subject in data:
        everyfoldlist = list()
        input_data = list()
        target_labels = list()
        [input_data.extend(data[subject]["session" + str(idx)]['input data']) for idx in session] # list:720 list:4 ndarray:3,875
        [target_labels.extend(labels[subject]["session" + str(idx)]) for idx in session] # list:720
        input_data = np.array(input_data) # ndarray:(720,4,3,875)
        target_labels = np.array(target_labels) # ndarray:(720,1)

        print("input_data shape is:", input_data.shape) #(720,4,3,875)
        print("target_labels shape is:", target_labels.shape) #(720,1)

        # 10 fold cross-validation
        count = 0
        for train_index, test_index in kf.split(input_data):
            count += 1
            # 720*11=7920 720-72=648 648*11=7128
            train_data, train_labels = arrange_data(input_data[train_index], target_labels[train_index]) # (2592,3,875,1) train_data:648,  train_labels:648
            test_data, test_labels = arrange_data(input_data[test_index], target_labels[test_index]) # test_data:72, test_labels:72 一共720个标签 测试集中1/10即72个标签

            size_y, size_x = train_data[0].shape[0:2]
            print("size_y is", size_y) # 3
            print("size_x is", size_x) # 875


            # 训练集和标签打乱顺序
            np.random.seed(120)
            np.random.shuffle(train_data)
            np.random.seed(120)
            np.random.shuffle(train_labels)


            # 测试集和标签打乱顺序
            np.random.seed(120)
            np.random.shuffle(test_data)
            np.random.seed(120)
            np.random.shuffle(test_labels)

            print("train_data shape is:", train_data.shape) #(7128, 90, 32, 1)
            print("train_labels shape is", train_labels.shape) #(7128,)

            #train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
            #test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

            # build model
            model = Shallow_Fusion_LSTM(size_y, size_x) # Shallow_Fusion1   Shallow_Fusion_LSTM

            # train the model
            lrate = LearningRateScheduler(step_decay)#learning rate 指数衰减
            epochs = 50  #50
            print("Training -----------------------------------")
            history = model.fit(train_data, train_labels, epochs=epochs, batch_size=32, validation_data=(test_data, test_labels), validation_freq=1,  shuffle=True, callbacks=[lrate]) #, callbacks=[lrate]  , callbacks=[lrate] , validation_data=(test_data, test_labels)
            # history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels), batch_size=32, validation_freq=1, callbacks=[lrate]) #shuffle=True,  #, callbacks=[lrate]
            # history = model.fit(train_data, train_labels, epochs=100, validation_split=0.2, batch_size=32, validation_freq=1) #history=model.fit   validation_data=(test_data, test_labels), validation_freq=1
            # Evaluate the model with the metrics we defined earlier
            print("\nTesting -----------------------------------")
            loss, accuracy = model.evaluate(test_data, test_labels)


            # 显示训练集和验证集的acc和loss曲线
            # model.fit训练过程中记录了训练集、测试集loss和准确率   用history.history提取出来
            acc = history.history['sparse_categorical_accuracy']
            val_acc = history.history['val_sparse_categorical_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            '''
            #画图
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
            #'''

            print(count, subject)
            print("test loss:", loss)
            print("test accuracy:", accuracy)
            everyfoldlist.append(accuracy)
        result.append(everyfoldlist)
    np.save('acc.npy', result)
    classification_acc = pd.DataFrame(result)
    return classification_acc
########################################################################################################################################################

if __name__ == "__main__":
    # data = np.load('data.npy', allow_pickle=True).item()
    # data = np.load('dataZnorm.npy', allow_pickle=True).item()
    # step1: 加载数据
    data = np.load('dataOri.npy', allow_pickle=True).item() #原始数据
    labels = np.load('labels.npy', allow_pickle=True).item()#原始标签
    sfreq = 250
    print('load data success!')
    # step2: 滤波
    data_bandpass = bandpass(data, 8, 30, 8, sfreq)  # 4, 32
    print('bandpass success!')
    print('start data process......')
    # step3: 处理
    data_processed, labels_processed = process(data_bandpass, labels, sfreq)
    print('data process success!')
    #step4: 交叉验证
    rst = run_classification(data_processed, labels_processed)
    print('////////////////////////////////////////////')
    # print('classification_accuracy: ', rst)
    rst['average'] = rst.mean(axis=1)
    print('average_accuracy: ', rst)
    print('////////////////////////////////////////////')
    rst.to_csv("BP_acc.csv", encoding="utf-8")