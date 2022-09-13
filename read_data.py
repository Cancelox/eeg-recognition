import numpy as np
import pandas as pd
import mne
import os
import re
from scipy.io import loadmat

class RawEEGData:
    def __init__(self, filename):
        raw = mne.io.read_raw_gdf(filename, preload=True, stim_channel=-1)
        event_type2idx = {276: 0, 277: 1, 768: 2, 769: 3, 770: 4, 781: 5, 783: 6, 1023: 7, 1077: 8, 1078: 9, 1079: 10,
                          1081: 11, 32766: 12}
        self.rawData = raw._data #size: 6*604803
        self.channel = raw._raw_extras[0]['ch_names']
        self.sample_freq = raw.info['sfreq']
        self.event = pd.DataFrame({
            "length":raw._raw_extras[0]['events'][0], #271个event
            "position": raw._raw_extras[0]['events'][1], #点的位置
            "event type": raw._raw_extras[0]['events'][2], #event的类型
            "event index": [event_type2idx[event_type] for event_type in raw._raw_extras[0]['events'][2]], #event的类型与event_type2idx定义的索引对应
            "duration": raw._raw_extras[0]['events'][4], #持续时间
            "CHN": raw._raw_extras[0]['events'][3]
        })
        #print("raw is:", raw)
        #print("ch_names is:", self.channel)
        #print("sfreq is:", self.sample_freq)
        #print("event is: \n", self.event)

    # print event type information of EEG data set
    @staticmethod #静态方法：可以不例化直接调用
    def print_type_info():
        print("EEG data set event information and index:")
        print("%12s\t%10s\t%30s" % ("Event Type", "Type Index", "Description"))
        print("%12d\t%10d\t%30s" % (276, 0, "Idling EEG (eyes open)"))
        print("%12d\t%10d\t%30s" % (277, 1, "Idling EEG (eyes closed"))
        print("%12d\t%10d\t%30s" % (768, 2, "Start of a trial"))
        print("%12d\t%10d\t%30s" % (769, 3, "Cue onset left (class 1)"))#data we want
        print("%12d\t%10d\t%30s" % (770, 4, "Cue onset right (class 2)"))#data we want
        print("%12d\t%10d\t%30s" % (781, 5, "BCI feedback (continuous"))
        print("%12d\t%10d\t%30s" % (783, 6, "Cue unknown"))
        print("%12d\t%10d\t%30s" % (1023, 7, "Rejected trial"))
        print("%12d\t%10d\t%30s" % (1077, 8, "Horizontal eye movement"))
        print("%12d\t%10d\t%30s" % (1078, 9, "Vertical eye movement"))
        print("%12d\t%10d\t%30s" % (1079, 10, "Eye rotation"))
        print("%12d\t%10d\t%30s" % (1081, 11, "Eye blinks"))
        print("%12d\t%10d\t%30s" % (32766, 12, "Start of a new run"))

# arrange data for training and test
def get_data(data_file_dir, labels_file_dir):
    #RawEEGData.print_type_info()
    sfreq = 250#采样频率
    data = dict()
    data_files = os.listdir(data_file_dir)#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for data_file in data_files:
        if not re.search(".*\.gdf", data_file):
            continue #跳过当前循环的剩余语句，然后继续进行下一轮循环。

        info = re.findall("B0([0-9])0([0-9])[TE]\.gdf", data_file)#re.findall(): 返回string中所有与pattern匹配的全部字符串,返回形式为数组
        # 提取所有符合要求的.gdf格式的文件
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = data_file_dir + "\\" + data_file

            raw_eeg_data = RawEEGData(filename)
            trial_event = raw_eeg_data.event[raw_eeg_data.event["event index"] == 2]#16行
            # print("trial_event is: \n", trial_event)
            session_data = dict()
            # iterrows()是在数据框中的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象。 event:行 event_data:列
            for event, event_data in trial_event.iterrows(): #event:索引 event_data:单条数据
                trial_data = raw_eeg_data.rawData[:, event_data["position"]:event_data["position"]+event_data["duration"]]
                for idx in range(len(raw_eeg_data.channel)):
                    if raw_eeg_data.channel[idx] not in session_data:
                        session_data[raw_eeg_data.channel[idx]] = list()
                    session_data[raw_eeg_data.channel[idx]].append(trial_data[idx])
            if subject not in data:
                data[subject] = dict()
            data[subject][session] = session_data
        except Exception as e:
            print(e)
            raise("invalid data file name")


    # read data file
    labels = dict()
    labels_files = os.listdir(labels_file_dir)
    for labels_file in labels_files:
        if not re.search(".*\.mat", labels_file):
            continue

        info = re.findall("B0([0-9])0([0-9])[TE]\.mat", labels_file)
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            filename = labels_file_dir + "\\" + labels_file
            session_label = loadmat(filename)
            session_label = session_label["classlabel"].astype(np.int8)
            if subject not in labels:
                labels[subject] = dict()
            labels[subject][session] = session_label
        except Exception as e:
            print(e)
            raise("invalid labels file name")

    return data, labels, sfreq

#归一化-1~1之间
def data_normalization(data):
    for subject in data:
        for session in data[subject]:
            for channel in data[subject][session]:
                for event in range(len(data[subject][session][channel])):
                    data[subject][session][channel][event] = data[subject][session][channel][event][:] / max(abs(data[subject][session][channel][event]))
    return data
#或
#Z-score normalization 0~1之间
# def data_Znormalization(data):
#     for subject in data:
#         for session in data[subject]:
#             for channel in data[subject][session]:
#                 for event in range(len(data[subject][session][channel])):
#                     mean = data[subject][session][channel][event].mean()
#                     sigma = data[subject][session][channel][event].std()
#                     dataZnorm = (data[subject][session][channel][event] - mean) / sigma
#                     data[subject][session][channel][event] = (dataZnorm - np.min(dataZnorm))/np.ptp(dataZnorm)
#                     #np.ptp()：最大值-最小值  把数据放到0~1之间
#     return data


if __name__ == "__main__":
    """
    filename = r"D:\Thesis2022\CI Competition IV\BCICIV_2b_gdf\B0101T.gdf"
    d = RawEEGData(filename)
    # print("event index is: \n", d.event['event index']) #所有的事件编号
    data = d.event[d.event['event index'] == 2] #取出编号为2 "Start of a trial"
    print("data is: \n", data)
    """

    data_src = r"D:\Thesis2022\BCI Competition IV\BCICIV_2b_gdf"
    labels_src = r"D:\Thesis2022\BCI Competition IV\true_labels_2b"
    data, labels, sfreq = get_data(data_src, labels_src)
    data = data_normalization(data) # 归一化-1~1之间
    # dataZ = data_Znormalization(data) # 归一化0~1之间 Z正则化
    np.save('dataOri.npy', data) # 原始数据
    np.save('data.npy', data) # 归一化-1~1之间
    # np.save('dataZnorm.npy', dataZ) # 归一化0~1之间 Z正则化
    np.save('labels.npy', labels)
    print("read_data over")