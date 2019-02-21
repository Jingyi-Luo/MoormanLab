import os
plt.rc('font', size=14)
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
#from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

#
working_dir = '/Users/ljyi/Desktop/capstone/capstone8/auto_encoder'
os.chdir(working_dir)

#==============================================================================
#                             autoencoder
#==============================================================================

RANDOM_SEED=42

x_tr = pd.read_csv('x_train.csv')
x_te = pd.read_csv('x_test.csv')
y_tr = pd.read_csv('y_train.csv')
y_te = pd.read_csv('y_test.csv')

ncol = x_tr.shape[1]  # 48
encoding_dim = 14

# input placeholder
input_layer = Input(shape=(ncol,))

# encoder layers 
encoder1 = Dense(28, activation='relu')(input_layer)
encoder2 = Dense(14, activation='relu')(encoder1)     

# decoder layers  
decoder1 = Dense(14, activation='relu')(encoder2)
decoder2 = Dense(ncol, activation='relu')(decoder1)    # activation='sigmoid'

#
autoencoder = Model(inputs=input_layer, outputs=decoder2)

# -------------------------------------------------------------
# a separate encoder model
encoder = Model(input_layer, encoder2)

# a separate decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

#encoder.summary()
#decoder.summary()
# ------------------------------------------------------------

# compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error') #(optimizer = 'adadelta', loss = 'binary_crossentropy') 
autoencoder.summary()
#Total params: 2,708
#Trainable params: 2,708
#Non-trainable params: 0

# save model
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=0, save_best_only=True)
# a_test  ??? Doesn't save successfully
#autoencoder_test = load_model('model.h5')

# export in a format that tensorboard understands
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# train autoencoder
nb_epoch = 10
batch_size = 100
autoencoder.fit(x_tr, x_tr, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(x_te, x_te),
                callbacks=[checkpointer, tensorboard])
                      # what is the role of x_tr and x_te. Does x_te get involved in trainning model? calculate loss function?
# - loss: 0.5860 - val_loss: 0.5286
# error distribution
x_te.shape    # 663304*48

# ----------------- extract compressed features -----------------
# get the encoded representation taining and test sets
encoded_tr = encoder.predict(x_tr)  # (68412, 14)
encoded_te = encoder.predict(x_te)  # (663304, 14)

# save to csv files
encoded_tr_df = pd.DataFrame(encoded_tr)
encoded_tr_df.to_csv('encoded_tr.csv', index=False)
encoded_te_df = pd.DataFrame(encoded_te)
encoded_te_df.to_csv('encoded_te.csv', index=False)
















































# -----------------------------------------------------------------------------
# ------------------------- mse and ROC curve ---------------------------------
#predictions = autoencoder.predict(x_te)
#predictions.shape   # 663304*48
#mse = np.mean(np.power(x_te-predictions, 2), axis=1)   # Length: 663304
#error_df =pd.DataFrame({'reconstruction_error':mse, 'y': y_te}) # # 663304*2
## error_df.describe()
##            reconstruction_error
##count         663304.000000
##mean               0.720468
##std                1.912979
##min                0.052035
##25%                0.379082
##50%                0.529522
##75%                0.784561
##max             1018.317202
#
## keep in mind the nature of dataset. ROC doesn't look very useful
## AUC
#fpr, tpr, thresholds = roc_curve(error_df.y, error_df.reconstruction_error)
#roc_auc = auc(fpr,tpr)   # 0.6726
#
## ROC curve from mse and AUC
#plt.title('Area Under Curve for AutoEncoder')
#plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.001, 1])
#plt.ylim([0, 1.001])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show();

# precision, recall
#precision, recall, th = precision_recall_curve(error_df.y, error_df.reconstruction_error)
#plt.plot(recall, precision, 'b', label='Precision-Recall curve')
#plt.title('Recall vs Precision')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.show()
#
## precision for different thresholds
#plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
#plt.title('Precision for different threshold values')
#plt.xlabel('Threshold')
#plt.ylabel('Precision')
#plt.show()
#
## recall for different thresholds
#plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
#plt.title('Recall for different threshold values')
#plt.xlabel('Reconstruction error')
#plt.ylabel('Recall')
#plt.show()
#
## prediction
#threshold = 2.9
#groups = error_df.groupby('y')
#fig, ax = plt.subplots()
#
#for name, group in groups:
#    print(group)
#    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
#            label= "Event" if name == "True" else "Non-event")
#ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
#ax.legend()
#plt.title("Reconstruction error for events")
#plt.ylabel("Reconstruction error")
#plt.xlabel("Data point index")
#plt.show();
# -----------------------------------------------------------------------------



# 1. reconstruct/represent the original data. Encoder and decoder: how does the model learn from the data?
# 2. encoder: force the model to learn the useful properties of the data and reduce dimension by limitting the number of
#    hidden units or add other constraints.
# 3. autoencoder use: 1) dimension reduciton (compressed representation of the input, similar to PCAs. Loose information(avoid)?); 
#    2) feature learning;
# 4. decoder: effectiveness?
# 5. 1)used the encodered features with reduced dimension to train model? 2) use the autoencodered/reconstructed feature
#    with the same dimension to train model? 3) a model itself?
# 6. An autoencoder neural network is an unsupervised learning algorithm that applied backpropagation, setting the target
#    values to be equal to the inputs. How to understand the equal?
#    Independent algorithm or need to work with other algorithm?
# 7. How to define/choose the hidden units (more than inputs, equal to inputs, or less than input)?
# 8. How to choose activation function? Related with input?
#    We would like the average activation of each hidden neuron j to be close to 0.05 (say). To satisfy this constraint, 
#    the hidden unit’s activations must mostly be near 0.
# 9. in almost all contexts where the term "autoencoder" is used, the compression and decompression functions are 
#    implemented with neural networks.
# 10.Autoencoders are data-specific, which means that they will only be able to compress data similar to what they have
#    been trained on. E.g. An autoencoder trained on pictures of faces would do a rather poor job of compressing pictures of trees.
# 11.I am curious about its application (areas and popularity)? Advantages?
# 12.Sparsity constraints? Deactive some units (why)?
#   encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
#    With this regularization, the model is less likely to overfit and can be trained longer)
# 13.





