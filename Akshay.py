#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import scipy
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics
from sklearn.model_selection import train_test_split


# In[ ]:





# In[8]:


eeg1 = pd.read_csv("train_eeg1.csv", index_col='Id')
eeg2 = pd.read_csv("train_eeg2.csv", index_col='Id')
emg = pd.read_csv("train_emg.csv", index_col='Id')
df_y = pd.read_csv("train_labels.csv", index_col='Id')

eeg1_t = pd.read_csv("test_eeg1.csv", index_col='Id')
eeg2_t = pd.read_csv("test_eeg2.csv", index_col='Id')
emg_t = pd.read_csv("test_emg.csv", index_col='Id')

eeg1.head()


# In[9]:




# convert to np arrays
np_eeg1 = np.array(eeg1)
np_eeg2 = np.array(eeg2)
np_emg = np.array(emg)
np_y = np.ravel(np.array(df_y))

np_eeg1_t = np.array(eeg1_t)
np_eeg2_t = np.array(eeg2_t)
np_emg_t = np.array(emg_t)

counter = Counter(np_y)
print(counter)

# upsample REM class (class 3) using resampling with replacement, while maintaining temporal consistency
rem_indices = [idx for idx, label in enumerate(np_y) if label == 3]
indices_to_resample = np.random.choice(rem_indices, size=counter[1] - counter[3])

repetitions = []
for i in range(len(np_y)):
    occurrences = np.count_nonzero(indices_to_resample == i)
    if (occurrences > 0):
        # resampling this index, so add it to repetitions (adding 1 to count the original)
        repetitions.append(occurrences + 1)
    else:
        # not resampling this index, so the repetition will only be 1 (i.e. it won't be duplicated)
        repetitions.append(1)

eeg1_res = np.repeat(np_eeg1, repeats=repetitions, axis=0)
eeg2_res = np.repeat(np_eeg2, repeats=repetitions, axis=0)
emg_res = np.repeat(np_emg, repeats=repetitions, axis=0)
y_res = np.repeat(np_y, repeats=repetitions, axis=0)

counter = Counter(y_res)
print(counter)

np_eeg1, np_eeg2, np_emg, np_y = eeg1_res, eeg2_res, emg_res, y_res


# In[ ]:


# transform data by concatenating data from 5 consecutive epochs together (use "padding" for epochs at start and end)

def concat_epochs(eeg1, eeg2, emg):
    num_epochs = len(eeg1)
    
    eeg1_transformed = []
    eeg2_transformed = []
    emg_transformed = []
    
    for i in range(0, num_epochs):
        if i == 0:
            epoch1, epoch2, epoch3, epoch4, epoch5 = 0, 0, 0, 1, 2
        elif i == 1:
            epoch1, epoch2, epoch3, epoch4, epoch5 = 0, 0, 1, 2, 3
        elif i == num_epochs - 2:
            epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i+1, i+1
        elif i == num_epochs - 1:
            epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i, i
        else:
            epoch1, epoch2, epoch3, epoch4, epoch5 = i-2, i-1, i, i+1, i+2
        
        eeg1_epochs = np.concatenate((eeg1[epoch1], eeg1[epoch2], eeg1[epoch3], eeg1[epoch4], eeg1[epoch5]))
        eeg1_transformed.append(eeg1_epochs)
    
        eeg2_epochs = np.concatenate((eeg2[epoch1], eeg2[epoch2], eeg2[epoch3], eeg2[epoch4], eeg2[epoch5]))
        eeg2_transformed.append(eeg2_epochs)
    
        emg_epochs = np.concatenate((emg[epoch1], emg[epoch2], emg[epoch3], emg[epoch4], emg[epoch5]))
        emg_transformed.append(emg_epochs)
    
    return np.array(eeg1_transformed), np.array(eeg2_transformed), np.array(emg_transformed)

eeg1_transf, eeg2_transf, emg_transf = concat_epochs(np_eeg1, np_eeg2, np_emg)  
np_eeg1, np_eeg2, np_emg = eeg1_transf, eeg2_transf, emg_transf

print("np_eeg1.shape: {}".format(np_eeg1.shape)) # should be (95361, (512 x 5)) = (95361, 2560)
print("np_eeg2.shape: {}".format(np_eeg2.shape)) # should be (95361, (512 x 5)) = (95361, 2560)
print("np_emg.shape: {}".format(np_emg.shape)) # should be (95361, (512 x 5)) = (95361, 2560)


# In[ ]:




# decrement each label by 1 to fit with sparse categorical cross entropy - we then increment the label by 1 in the predictions
# i.e. we train on labels [0, 1, 2], which map to [1, 2, 3] in our predictions
y_dec = [label - 1 for label in np_y]
np_y = np.array(y_dec)

eeg1_train, eeg1_validation, y_train, y_validation = train_test_split(np_eeg1, np_y, test_size=0.33, shuffle=False)
eeg2_train, eeg2_validation, _, _ = train_test_split(np_eeg2, np_y, test_size=0.33, shuffle=False)
emg_train, emg_validation, _, _ = train_test_split(np_emg, np_y, test_size=0.33, shuffle=False)

X_train = np.dstack((eeg1_train, eeg2_train, emg_train))
print("X_train.shape: {}".format(X_train.shape))

X_validation = np.dstack((eeg1_validation, eeg2_validation, emg_validation))
print("X_validation.shape: {}".format(X_validation.shape))

# neural net
model = models.Sequential()
model.add(layers.BatchNormalization())
# try with 128 filters (to match sampling frequency)
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=1, activation='relu')) # layer 1
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=2, activation='relu')) # layer 2
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=1, activation='relu')) # layer 3
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=2, activation='relu')) # layer 4
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=1, activation='relu')) # layer 5
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=2, activation='relu')) # layer 6
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=1, activation='relu')) # layer 7
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=128, kernel_size=(5,), strides=2, activation='relu')) # layer 8
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dropout(0.2, input_shape=(128,)))
model.add(layers.Dense(80, activation='relu')) # layer 9, fc1
model.add(layers.Dense(3, activation='softmax')) # layer 10, fc2



starter_learning_rate = 0.0
end_learning_rate = 0.00128*256
decay_steps = 5


learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = starter_learning_rate,
    decay_steps = decay_steps,
    end_learning_rate = end_learning_rate,
    power=1)

learning_rate_fn2 = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = end_learning_rate,
    decay_steps = decay_steps,
    end_learning_rate = starter_learning_rate,
    power=1)

#def decayed_learning_rate(step):
#    starter_learning_rate = 0.0
#    end_learning_rate = 0.00128*256
#    decay_steps = 5
#    step  = min(step, decay_steps)
#    res = ((initial_learning_rate - end_learning_rate) * ((1 - (step/decay_steps) )**(power))) + end_learning_rate
#    return res


#def scheduler(epoch, lr):
#  if epoch < 5:
#    return lr
#  else:
#    return lr * tf.math.exp(-0.1)



opt = tf.keras.optimizers.RMSprop(
                  learning_rate=learning_rate_fn, rho = 0.99)

#opt = tf.keras.optimizers.RMSprop(learning_rate=1e-2)

model.compile(optimizer=opt,
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#K.set_value(model.optimizer.learning_rate, 0.001)


model.fit(X_train, y_train, epochs=5, batch_size=256)

opt.learning_rate.assign(end_learning_rate)
model.fit(X_train, y_train, epochs=10, batch_size=256)

opt.learning_rate.assign(learning_rate_fn2)
model.fit(X_train, y_train, epochs=5, batch_size=256)


model.evaluate(X_validation, y_validation, verbose=2)


# In[ ]:


# apply concatenation transformation to test data
eeg1_t_transf, eeg2_t_transf, emg_t_transf = concat_epochs(np_eeg1_t, np_eeg2_t, np_emg_t)  
np_eeg1_t, np_eeg2_t, np_emg_t = eeg1_t_transf, eeg2_t_transf, emg_t_transf

X_test = np.dstack((np_eeg1_t, np_eeg2_t, np_emg_t))
print("X_test.shape: {}".format(X_test.shape))

# predict
predictions = model.predict(X_test)
result =predictions

df_r = pd.DataFrame(result, index=eeg1_t.index.astype(int))
df_out = pd.concat([df_r], axis=1, sort=False)
df_out.columns = df_y.columns

df_out.to_csv('preds.csv', index=True, header=True, float_format='%.3f')


# In[ ]:





# In[ ]:





# In[ ]:




