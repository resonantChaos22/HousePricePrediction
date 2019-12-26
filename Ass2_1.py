#IMPORTS
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("https://raw.githubusercontent.com/someshsingh22/CTE-Intro-To-Machine-Learning/master/Assignment_2/housepricedata.csv").astype('int32')
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#defining variables
input_size = 10
hidden_size = 5

num_classes = 2
num_epochs = 500


learning_rate = 0.001


#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#importing dataset
df = pd.read_csv("https://raw.githubusercontent.com/someshsingh22/CTE-Intro-To-Machine-Learning/master/Assignment_2/housepricedata.csv").astype('int32')


#creating train and test data
msk = np.random.rand(len(df)) < 0.75
train = df[msk]
test = df[~msk]

#creating custom datsaset
class HouseData(Dataset):
    
    def __init__(self, dataframe):
        self.samples = train
        
        
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        xdata = self.samples.iloc[:, :-1].values[idx]
        ydata = self.samples.iloc[:, -1].values[idx]
        sample = {'xdata' : xdata, 'ydata' : ydata}
        return sample
 
#creating dataloaders

train_set = HouseData(train)
test_set = HouseData(test)

train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)
#creating neural network
class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU6()
        self.BatchNorm = nn.BatchNorm1d(5)
        #self.Drop = nn.Dropout(0.001)                 Dropout is of no use in this case
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        #out = self.Drop(out)
        out = self.BatchNorm(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
best_model = NeuralNet(input_size, hidden_size, num_classes).to(device)
leastLoss = 5

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#training
best_Accuracy = 0

for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        correct = 0
        total = 0
        #importing data to be trained
        xdata, ydata = Variable(sample['xdata']), Variable(sample['ydata'])
        xdata = xdata.to(device)
        ydata = ydata.to(device)
        xdata = xdata.float()
        ydata = ydata.long()
        
        #calculating loss
        output = model(xdata)
        loss = criterion(output, ydata)
        
        #to save the best model
        _, predicted0 = torch.max(output.data, 1)
        total += ydata.size(0)
        correct += (predicted0==ydata).sum().item()
        accuracy = (correct/total)*100

        
        if accuracy>best_Accuracy:
            t = accuracy
            best_Accuracy = t
            best_model = model
            #torch.save(best_model, 'bestmodel.pt')
            
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Epoch {}/{} and loss is {}'.format(epoch+1, num_epochs, loss.item()))
        

#testing model
#new_model = torch.load('bestmodel.pt')
with torch.no_grad():
    
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    for i, sample in enumerate(test_loader):
        #importing data to be trained
        xdata1, ydata1 = Variable(sample['xdata']), Variable(sample['ydata'])
        xdata1 = xdata1.to(device).float()
        ydata1 = ydata1.to(device).long()
        #output = new_model(xdata1)
        output1 = model(xdata1)
        output2 = best_model(xdata1)
       #_, predicted = torch.max(output.data, 1)
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        total += ydata1.size(0)
        #correct += (predicted==ydata1).sum().item()
        correct1 += (predicted1==ydata1).sum().item()
        correct2 += (predicted2==ydata1).sum().item()
    
    #print('Accuracy - {}%'.format((correct/total)*100))
    print('Accuracy - {}%'.format((correct1/total)*100))
    print('Accuracy - {}%'.format((correct2/total)*100))
    
test_submit=pd.read_csv("https://raw.githubusercontent.com/someshsingh22/CTE-Intro-To-Machine-Learning/master/Assignment_2/housepricedata_Test.csv")
submit_predictions = best_model(Variable(torch.tensor(test_submit.values).float()))
_, submit = torch.max(submit_predictions.data, 1)
m = pd.DataFrame(submit)
m.to_csv('submit.csv')
   
#torch.save(model.state_dict(), 'model.ckpt')
    
        


 