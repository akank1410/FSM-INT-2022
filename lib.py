import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import re
import glob 
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import io
import urllib, base64
import time


from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic

def files_merge(root):

    #bearing = re.findall('Bearing+[\w.]+',root)[0]
    # PHASE1 : Files Reading Phase
    print("\nEntering Files Reading Phase : Phase1")
    s = time.time()
    files = sorted(glob.glob(os.path.join(root,'acc_*.csv')))
    e = time.time()
    print("File Reading phase Successful : Phase1 - ",  (e-s)/60)

    print(len(files))

    # PHASE2 : File Merging phase
    print("\nEntering File Merging phase : Phase2")
    s = time.time()
    merged_files_csv = merge_tocsv(files)
    e = time.time()
    print("File Merging phase Successful : Phase2 - ",  (e-s)/60)
    #print(merged_files_csv)

    # PHASE3 : File to DataFrame phase
    print("\nEntering File to DataFrame phase : Phase3")
    s = time.time()
    header_col = ['hour', 'minute', 'second', 'microsecond',  'horiz accel',  'vert accel']
    df = pd.read_csv(merged_files_csv, names= header_col)
    os.remove(merged_files_csv)
    e = time.time()
    print("File to DataFrame phase Successful : Phase3 - ",  (e-s)/60)

    for r,d,lis in os.walk(root):
      files = lis
     

    for file in files:
      p = os.path.join(root,file)
      os.remove(p)

    return df


def merge_tocsv(file_lis):
  
    dir = "static\\processing_files" 
    
    csv_file = os.path.join(dir,"file.csv")

    with open(csv_file, 'w') as wr:
        for file in file_lis:
            with open(file , 'r') as read:
              for line in read:
                wr.write(line)

            os.remove(file)

    return csv_file
            
def load_file(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

def df_row_ind_to_data_range(ind):
   
    # Variables
    DATA_POINTS_PER_FILE = 2560

    return (DATA_POINTS_PER_FILE*ind, DATA_POINTS_PER_FILE*(ind+1))

def extract_feature_image(df,ind, feature_name='horiz accel'):

    #Variables
    DATA_POINTS_PER_FILE = 2560
    WIN_SIZE = 20
    WAVELET_TYPE = 'morl'

    data_range = df_row_ind_to_data_range(ind)
    data = df[feature_name].values[data_range[0]:data_range[1]]
    # use window to process(= prepare, develop) 1d signal
    data = np.array([np.mean(data[i:i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])  
    # perform cwt on 1d data
    coef, _ = pywt.cwt(data, np.linspace(1,128,128), WAVELET_TYPE)  
    # transform to power and apply logarithm ?!
    coef = np.log2(coef**2+0.001) 
    # normalize coef
    coef = (coef - coef.min())/(coef.max() - coef.min()) 
    return coef

def signal_processing(df):
    no_of_rows = df.shape[0]
    DATA_POINTS_PER_FILE = 2560
    no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
    no_of_samples = 5
    fig, ax = plt.subplots(2, no_of_samples, figsize=[20,8])
    ax[0,0].set_ylabel('horiz accel features image')
    ax[1,0].set_ylabel('vert accel features image')

    # dividing the feature images into 5 samples
    for i,p in enumerate(np.linspace(0,1,no_of_samples)):
      ind = int( (no_of_files-1)*p)

    # extracting and plotting horizontal accelration feature images for 5 samples
      coef = extract_feature_image(df,ind, feature_name='horiz accel') 
      ax[0,i].set_title('{0:.2f}'.format(p))
      im = ax[0,i].imshow(coef, cmap = 'coolwarm')
      fig.colorbar(im, ax = ax[0,i], fraction=0.046, pad=0.04)

    # extracting and plotting vertical accleration feature images for 5 samples
      coef = extract_feature_image(df,ind, feature_name='vert accel')
      ax[1,i].set_title('{0:.2f}'.format(p))
      im = ax[1,i].imshow(coef, cmap='coolwarm')
      fig.colorbar(im, ax = ax[1,i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri  
    return html

#PHASE 4 : 1d to 2d time-frequency extraction
def extract_2d_feature(df):

    no_of_rows = df.shape[0]
    DATA_POINTS_PER_FILE = 2560
    no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
  
    data = {'x': [], 'y': []}
    for i in range(0, no_of_files):
        coef_h = extract_feature_image(df,i, feature_name='horiz accel') # size = (128,128)
        coef_v = extract_feature_image(df,i, feature_name='vert accel')  # size = (128,128)

        x_ = np.array( [coef_h, coef_v] ) # size = (2,128,128)
        y_ = i/(no_of_files - 1) 
        data['x'].append(x_)
        data['y'].append(y_)

    data['x'] = np.array(data['x'])
    data['y'] = np.array(data['y'])

    assert data['x'].shape == (no_of_files, 2, 128, 128)

    #address of file converted from 1d to 2d
    return  data

#****_______________________________________________****

# PHASE5 
class PHMTestDataset_Sequential(Dataset):
    """PHM data set where each item is a sequence"""
    def __init__(self, dataset='', seq_len=5):
       
        self.data = dataset
        self.seq_len = seq_len
    
    def __len__(self):
        return self.data['x'].shape[0]-self.seq_len+1
    
    def __getitem__(self, i):
        sample = {'x': torch.from_numpy(self.data['x'][i:i+self.seq_len])}
        return sample

#****_______________________________________________****
# CNN + LSTM Model

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', batch_norm=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    nn.init.xavier_uniform_(conv.weight)
    relu = nn.ReLU()
    if batch_norm:
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            relu
        )
    else:
        return nn.Sequential(
            conv,
            relu
        )

class CNN_CWT_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn_relu(2, 16, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = conv_bn_relu(16, 32, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = conv_bn_relu(32, 64, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = conv_bn_relu(64, 128, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(p=0.2)
 
    def forward(self, x):
        # input shape = [Nx2x128x128]
        x = self.conv1(x) # [Nx16x128x128]
        x = self.pool1(x) # [Nx16x64x64]
        x = self.conv2(x) # [Nx32x64x64]
        x = self.pool2(x) # [Nx32x32x32]
        x = self.conv3(x) # [Nx64x32x32]
        x = self.pool3(x) # [Nx64x16x16]
        x = self.conv4(x) # [Nx128x16x16]
        x = self.pool4(x) # [Nx128x8x8]
        x = self.flatten(x) # [Nx8192] {128*8*8=8192} (N => batch size, 128 => no. of channels, 8*8 => height of image*width of image)
        x = self.fc1(x) # [Nx256] 
        # x = self.dropout1(x) # apply dropout (Dropout is much harder to implement in LSTM)
        x = nn.ReLU()(x) # apply ReLU activation
        x = self.fc2(x) # [Nx128]
        x = nn.ReLU()(x) # apply ReLU activation
        return x

class CNN_LSTM_FP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNN_CWT_Encoder()
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # input shape = [N x l x 2 x 128 x 128] Here, N - batch size, l - sequence length (i.e. SEQ_LEN = 5),  2 - no. of channels or no. of filters, 
                                                      # 128 * 128 - height of an image * width of an image
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size*seq_len, C, H, W) # transform input of shape [N x l x 2 x 128 x 128] into input of shape [(Nxl) x 2 x 128 x 128]. basically,
                                                # converting(= transforming) into sequences. [(Nxl) x 2 x 128 x 128] - transformed input sequence
        x = self.encoder(x) # pass transformed input sequence through CNN Encoder, CNN Encoder converts the image input data sequence of shape 
                            # [(Nxl) x 2 x 128 x 128] into linear vector sequence by flatenning, output feature vector sequence shape = [(Nxl) x 128]
        x = x.view(batch_size, seq_len, -1) # transform encoded feature vector sequence into time distributed(= shared, alloted, assigned) input as required by
                                            # LSTM unit or LSTM cell
        x, _ = self.lstm1(x) # pass transformed encoded feature vector sequence through LSTM unit or LSTM cell, _ variable contains the hidden layers or hidden
                             # states of LSTM, we don't require those hidden layers in our implementation, therefore just stored in _ variable, and if we want
                             # we can initialize(= activate, start) hidden states from _ variable, here hidden states are the array of zeroes
        x = self.fc(x[:,-1,:]) # pass last vector sequence(i.e. output vector sequence of LSTM unit at last time step) through fully connected network layer
        x = nn.Sigmoid()(x)
        return x    

#****_______________________________________________****

#Applying Inference
#Inference - Inference refers to the process of using a trained machine learning model to make a prediction. Basically inference means validating the trained model
# PHASE8
def model_inference_helper(model, dataloader,device):
    results = {'predictions':[]}
    model.eval()
    for i, batch in enumerate(dataloader):
        x = batch['x'].to(device, dtype=torch.float)

        with torch.no_grad():
            y_prediction = model(x)

        if y_prediction.size(0)>1:
            results['predictions'] += y_prediction.cpu().squeeze().tolist()
        elif y_prediction.size(0)==1:
            results['predictions'].append(y_prediction.cpu().squeeze().tolist())
    return results

#****_______________________________________________****

# Output Plots

def output1(df):

  no_of_rows = df.shape[0]
  plt.figure(figsize=(18, 7))
  plt.subplot(1,2,1)
  plt.plot(range(no_of_rows), df['horiz accel'])
  plt.legend(['horiz accel'])

  plt.subplot(1,2,2)
  plt.plot(range(no_of_rows), df['vert accel'], 'r')
  plt.legend(['vert accel'])
  fig = plt.gcf()

  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())

  uri = 'data:image/png;base64,' + urllib.parse.quote(string)
  html = '<img src = "%s"/>' % uri  
  return html

def output2(results):
  plt.figure(figsize=(10, 7))
  plt.scatter(range(len(results['predictions'])), results['predictions'], c = 'r', marker='.', label='predicted values')
  fig = plt.gcf()

  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())

  uri = 'data:image/png;base64,' + urllib.parse.quote(string)
  html = '<img src = "%s"/>' % uri  
  return html

def output3(results):
  X = np.arange(len(results['predictions'])).reshape(-1,1)
  y = np.array(results['predictions']).reshape(-1, 1)
  reg = LinearRegression().fit(X, y)

  X_test = np.linspace(0, 5000, 10000).reshape(-1, 1)
  y_test = reg.predict(X_test)
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
  ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
  ax.plot(X_test, y_test)
  ax.plot(X_test, [0.9]*len(X_test), 'k')
  fig = plt.gcf()

  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())

  uri = 'data:image/png;base64,' + urllib.parse.quote(string)
  html = '<img src = "%s"/>' % uri  
  address = "static\\images\\img1.jpg"
  fig.savefig(address)


def output4(results):
  X = np.arange(len(results['predictions'])).reshape(-1,1)
  y = np.array(results['predictions']).reshape(-1, 1)
  gp_kernel = RBF(0.01)+RationalQuadratic(0.1)

  gpr = GaussianProcessRegressor(kernel=gp_kernel)
  gpr.fit(X, y)
  
  X_test = np.linspace(0, 5000, 10000).reshape(-1, 1)
  y_test = gpr.predict(X_test)
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[11.5, 8])
  ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
  ax.plot(X_test, y_test, 'r')
  ax.plot(X_test, [0.9]*len(X_test), 'k')

  fig = plt.gcf()
  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())

  uri = 'data:image/png;base64,' + urllib.parse.quote(string)
  html = '<img src = "%s"/>' % uri  

  address = "static\\images\\img2.jpg"
  fig.savefig(address)

  

# connecting device & model calling
# PHASE7
def model_Execution():
  # Switching from CPU to GPU
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

  # Connecting(=linking) our model to the device(GPU) to train the model on GPU
  model = CNN_LSTM_FP().to(device)

  # Loading the Model
  model.load_state_dict(torch.load('d:\\my_flask\\cnn_lstm_model3.pth',map_location=torch.device('cpu')))

  return model,device

def prediction(path):
  # input(path)

  # PHASE1, PHASE2, PHASE3
  df = files_merge(path)

  # PHASE4
  # PHASE 4 : 1d to 2d time-frequency feature extraction
  print("\nEntering 1d to 2d time-frequency feature extraction phase : Phase4")
  s = time.time()
  file_1d_2d = extract_2d_feature(df)
  e = time.time()
  print("1d to 2d time-frequency feature extraction phase Successful: Phase4",  e-s)

  
  # PHASE5
  print("\nEntering Data Preparation Phase for the model : Phase5")
  s = time.time()
  test_dataset = PHMTestDataset_Sequential(dataset=file_1d_2d)
  e = time.time()
  print("Entering Data Preparation Phase for the model Successful : Phase5 - ",  (e-s)/60)

  Batch_size = 16
  
  # here dataloader() that loads batch of 16 samples(test data sequences) at a time 
  # PHASE 6
  print("\nEntering DataLoader Phase : Phase6")
  s = time.time()
  dataloaders = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1) 
  e = time.time()
  print("DataLoader Phase Successful : Phase6 - ", (e-s)/60) 

  # PHASE 7
  print("\nEntering Model & Device Setup Phase : Phase7")
  s = time.time()
  model,device = model_Execution()
  e = time.time()
  print("Model & Device Setup Phase Successful : Phase7 - ", (e-s)/60)

  # PHASE 8 
  print("\nEntering Prediction Phase : Phase8")
  s = time.time()
  results = model_inference_helper(model,dataloaders,device)
  e = time.time()
  print("Prediction Phase Successful : Phase8 - ", (e-s)/60)
  
  #Plots
  # PHASE 9
  s = time.time()
  print("\n Entering Result display Phase : Phase9")
  #h1 = output13(df) 
  #h2 = signal_processing(df)
  #h3 = output2(results)
  h4 = output3(results)
  h5 = output4(results)   
  e = time.time()
  print("Result display Phase Successful: Phase9 - ", (e-s/60))

  return (results)