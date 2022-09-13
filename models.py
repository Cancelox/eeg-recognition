import tensorflow as tf
from keras.layers import Dense, AveragePooling2D
from tensorflow import keras
from tensorflow.keras.layers import Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
#from tensorflow.contrib.layers import xavier_initializer

############################################################################################
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))
############################################################################################

############################################################################################
# -------------------------------------ShallowFusion---------------------------------------#
############################################################################################
#所有层的介绍：https://blog.csdn.net/zimiao552147572/article/details/104438651
##Shallow_Fusion1：多通道
def Shallow_Fusion1(Chans, Samples):
    nb_classes = 2
    dropoutRate = 0.5

    input_main = Input((Chans, Samples, 1))
    # 1
    block1 = Conv2D(90, (1, 65),
                    strides=(1, 3),
                    use_bias=True,
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=regularizers.l2(0.02),
                    # activation='relu'
                    )(input_main)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1) # epsilon=1e-05, momentum=0.1

    # 3
    block1 = Conv2D(90, (Chans, 1),
                    strides=(1, 1),
                    use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=regularizers.l2(0.02),
                    # activation='relu'
                    )(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1) # epsilon=1e-05, momentum=0.1
    #block1 = Activation('relu')(block1)
    block1 = Activation(square)(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block1 = Permute((2, 1, 3))(block1)
    block1 = Reshape((block1.shape[1], block1.shape[3]))(block1)
    #'''
    #LSTM
    # 1
    block1 = LSTM(64, recurrent_activation = 'sigmoid', return_sequences=True, dropout=0.2, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.02))(block1) #dropout=0.2, recurrent_dropout=0.2,, kernel_regularizer=regularizers.l2(0.08)
    block1 = Dropout(dropoutRate)(block1)
    # 2
    block1 = LSTM(64, recurrent_activation='sigmoid', return_sequences=True, dropout=0.2, recurrent_dropout=0.1,kernel_regularizer=regularizers.l2(0.02))(block1)
    # 3
    #block1 = LSTM(64, recurrent_activation='sigmoid', return_sequences=True, dropout=0.2, recurrent_dropout=0.1, kernel_regularizer=regularizers.l2(0.02))(block1)
    #block1 = Dropout(dropoutRate)(block1)
    #'''
    flatten1 = Flatten()(block1)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten1)  # , kernel_constraint=max_norm(0.5) , kernel_regularizer=regularizers.l2(0.1) , use_bias=True
    softmax = Activation('softmax')(dense)

    model = keras.models.Model(inputs=input_main, outputs=softmax)
    #model.summary()
    learningRate = 0.001
    # momentum = 0.9
    # decay_rate = 0.0
    sgd = tf.keras.optimizers.SGD(lr=learningRate, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer= sgd, # 'adam'
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy']
                  )
    return model

############################################################################################
def Shallow_Fusion_LSTM(Chans, Samples):#GRU搭建模型：https://blog.csdn.net/fendouaini/article/details/80272320
    # start the model
    nb_classes = 2
    dropoutRate = 0.5
    input_main = Input((Chans, Samples, 1))

    #block1
    block1 = Conv2D(45, (1, 65),#调
                    strides=(1, 3),
                    use_bias=False,
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=regularizers.l2(0.01),
                    #activation='relu'
                    )(input_main)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1 = Conv2D(45, (Chans, 1),
                    strides=(1, 1),
                    use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                    kernel_regularizer=regularizers.l2(0.01),
                    #activation='relu'
                    )(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    #block1 = Activation('relu')(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 6), strides=(1, 6))(block1)  #AveragePooling2D
    # block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block1 = Permute((2, 1, 3))(block1)# 可以同时多次交换tensor的维度 示例：b = a.permute(0,2 ,1) 将a的维度索引1和维度索引2交换位置
    #block1 = tf.reshape(block1, (int(block1.shape[0]), int(block1.shape[1]), int(block1.shape[2]*block1.shape[3]))) #tf.shape(block1)[0],
    block1 = Reshape((block1.shape[1], block1.shape[3]))(block1)
    #GRU
    #block1 = GRU(32, return_sequences=True)(block1)
    #block1 = GRU(32, return_sequences=True)(block1)
    block1 = GRU(32, return_sequences=True)(block1)
    block1 = GRU(32, return_sequences=False)(block1) # 最后一次用false
    # GRU: https://blog.csdn.net/weixin_44791964/article/details/104011262

    flatten1 = Flatten()(block1)

    #dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5) , kernel_regularizer=regularizers.l2(0.1) , use_bias=True)(flatten1)
    softmax = Activation('softmax')(dense)

    model = keras.models.Model(input_main, softmax)
    #model.summary()
    '''
    model.compile(optimizer=tf.keras.optimizers.Adam(), # 默认lr=0.001
                  #optimizer='adam',
                  loss='sparse_categorical_crossentropy', # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'] # acc
                  )
    #'''
    #'''
    learningRate = 0.001
    #sgd = tf.keras.optimizers.SGD(lr=learningRate, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(#optimizer=sgd,
                  optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy']
                  )
    #'''
    return model

