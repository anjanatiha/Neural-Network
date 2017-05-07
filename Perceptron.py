
# coding: utf-8

# In[1]:

import random
import pandas as pd
import numpy as np
import datetime
import time

def file_read():
    data1 = pd.read_csv("d-10.csv")
    data2 = pd.read_csv("d-100.csv")
    data3 = pd.read_csv("d-500.csv")
    return data1, data2, data3


# In[2]:

data1, data2, data3 = file_read()


# In[3]:

def bias_add(train_feature_matrix, bias_val):
    bias=np.zeros((train_feature_matrix.shape[0],1))
    for i in range(train_feature_matrix.shape[0]):
        bias[i]=bias_val
    train_feature_matrix = np.hstack((train_feature_matrix, bias))    
    return train_feature_matrix

def train_test_sep(data):
    train_target = data['C']   
    train_feature = data.drop('C', axis=1)
    return train_feature, train_target

def train_test_matrix(train_feature, train_target):
    train_feature_matrix = np.asarray(train_feature)
    train_target_matrix = np.asarray(train_target)
    train_feature_matrix = bias_add(train_feature_matrix, 1)
    
    return train_feature_matrix, train_target_matrix

def feature_len(data_train_matrix):
    return data_train_matrix.shape[1]

def row_len(data_train_matrix):
    return data_train_matrix.shape[0]

def feature_decode(column_value_array):
    length_row = len(column_value_array)
    column_value_array_new = np.empty([length_row, 1])
    for i in range(len(column_value_array)):
        if column_value_array[i] == 'NS':
            column_value_array_new[i] = -1
        if column_value_array[i] == 'S':
            column_value_array_new[i] = 1
    return column_value_array_new


# In[4]:

train_feature1, train_target1 = train_test_sep(data1)
train_feature_matrix1, train_target_matrix1 = train_test_matrix(train_feature1, train_target1)

train_feature2, train_target2 = train_test_sep(data2)
train_feature_matrix2, train_target_matrix2 = train_test_matrix(train_feature2, train_target2)
train_target_matrix2 = feature_decode(train_target_matrix2)


train_feature3, train_target3 = train_test_sep(data3)
train_feature_matrix3, train_target_matrix3 = train_test_matrix(train_feature3, train_target3)

feature_len1 = feature_len(train_feature_matrix1)
feature_len2 = feature_len(train_feature_matrix2)
feature_len3 = feature_len(train_feature_matrix3)

print(data2.shape)


# In[5]:

def rand_gen(start, end):
    return random.uniform(start, end)

def rand_gen_num(start, end, number):
    rand_num = []
    for i in range(number):
        rand_num.append(rand_gen(start, end))
    return rand_num

def activation(output, activation_value):
    if output>=activation_value:
        return 1.0
    else:
        return -1.0
    
def output_gen(feature_val, weight, activation_val):
    output_val = 0
    for i in range(len(feature_val)):
        output_val = output_val + feature_val[i]*weight[i]
    output_val = activation(output_val, activation_val)
    return output_val

def error_calc(target_val, pers_output):
    return target_val - pers_output

def weight_adjust(feature_val, error, learning_rate):
    weight_adjust = []
    for i in range(len(feature_val)):
        weight_adjust.append(feature_val[i]*learning_rate*error)
    return weight_adjust

def new_weight(current_weight, adjusted_weight):
    new_weight = []
    for i in range(len(current_weight)):
        new_weight.append(current_weight[i] + adjusted_weight[i])
    return new_weight

def predict_perceptron(train_feature_matrix, weight_array, activation_val):
    predicted_output = []
    for i in range(len(train_feature_matrix)):
        row = train_feature_matrix[i]
        row_output = output_gen(row, weight_array, activation_val)
        predicted_output.append(row_output)
    return predicted_output
    

def prediction_compare(train_target_matrix, predicted_target_matrix):
    for i in range(len(train_target_matrix)):
        print("Original: ", train_target_matrix[i], " Predicted: ", predicted_target_matrix[i], "\n")
    print("\n\n All comparision printed\n\n")
    
def error_percentile(train_target_matrix, predicted_target_matrix):
    total = 0
    misclassified = 0
    for i in range(len(train_target_matrix)):
        if train_target_matrix[i] == predicted_target_matrix[i]:
            total = total + 1
        else:
            misclassified = misclassified + 1
            total = total + 1
    error_percentile = (misclassified/total)*100
    return error_percentile
def textwrite(msg, filename):
    text_file = open(filename, "w")
    text_file.write(msg)
    text_file.close()
def save_np_text(msg, filename):
    np.savetxt(filename, msg, delimiter=',')

def timespend():
    start = time.time()
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
def cur_time():
    return str(datetime.timedelta(seconds=666))


# In[6]:

def perceptron_n_iter(train_feature_matrix, train_target_matrix, feature_len, learning_rate, bias, n_iter, activation_val):
    initial_weight = rand_gen_num(0, 1, feature_len)
    current_weight = initial_weight
    for i in range(n_iter):
        print (time.strftime("%H:%M:%S"))
        print("iter:", i, "\n")
        if i==100 or i == 500 or i == 1000:
            print(i," :model run completed\n")
            predicted_value = predict_perceptron(train_feature_matrix, current_weight, activation_val)
            print(i," :iter predictioon complete\n")
            err_percentile = error_percentile(train_target_matrix, predicted_value)
            error_msg = str(err_percentile)
            print(i, ":th iteration error percentile :", err_percentile)
            if i==100:
                text_file = open("hundred.txt", "w")
                text_file.write(error_msg)
                text_file.close()
            if i==500:
                text_file = open("fivehundred.txt", "w")
                text_file.write(error_msg)
                text_file.close()
            if i==1000:
                text_file = open("thousand.txt", "w")
                text_file.write(error_msg)
                text_file.close()
            current_weight = perceptron(train_feature_matrix, train_target_matrix, current_weight, learning_rate, bias, activation_val)
        
        current_weight = perceptron(train_feature_matrix, train_target_matrix, current_weight, learning_rate, bias, activation_val)

    final_weight_array = current_weight
    return final_weight_array

def perceptron(train_feature_matrix, train_target_matrix, current_weight, learning_rate, bias, activation_val):
    new_weight_array = current_weight
    for i in range(len(train_feature_matrix)):
        output=0
        row = train_feature_matrix[i]
        output = output_gen(row, current_weight, activation_val)
        error = error_calc(train_target_matrix[i], output)
        weight_adjust_array = weight_adjust(row, error, learning_rate)
        new_weight_array = new_weight(new_weight_array, weight_adjust_array)
    return new_weight_array


# In[ ]:

train_feature_matrix = train_feature_matrix1
train_target_matrix = train_target_matrix1
feature_len = feature_len1
learning_rate = 0.001
bias=1
n_iter=1000
activation_val= 1
weight_array = perceptron_n_iter(train_feature_matrix, train_target_matrix, feature_len, learning_rate, bias, n_iter, activation_val)


# In[ ]:

train_feature_matrix2 = train_feature_matrix2
train_target_matrix2 = train_target_matrix2
feature_len2 = feature_len2
learning_rate2 = 0.01
bias2=1
n_iter2=1002
activation_val2= 1
weight_array2 = perceptron_n_iter(train_feature_matrix2, train_target_matrix2, feature_len2, learning_rate2, bias2, n_iter2, activation_val2)


# In[ ]:

train_feature_matrix3 = train_feature_matrix3
train_target_matrix3 = train_target_matrix3
feature_len3 = feature_len3
learning_rate3 = 0.01
bias3=1
n_iter3=1000
activation_val3= 1
weight_array3 = perceptron_n_iter(train_feature_matrix3, train_target_matrix3, feature_len3, learning_rate3, bias3, n_iter3, activation_val3)


# In[ ]:



