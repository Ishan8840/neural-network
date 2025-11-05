import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, input=4, h1=8, h2=9, output=3):
        super().__init__()
        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
my_df['species'] = my_df['species'].map(label_map)

X = my_df.drop('species', axis = 1).values
y = my_df['species'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100

losses = []

for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    y_pred_classes = y_pred.argmax(axis=1)
    correct = (y_pred_classes == y_train).sum()
    accuracy = correct / y_train.shape[0]

    if epoch % 10 == 0:
         print(f'loss: {loss}, accuracy: {accuracy}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.show()



with torch.no_grad():
    y_eval = model(X_test)

    y_eval_classes = y_eval.argmax(axis=1)
    correct = (y_eval_classes == y_test).sum()
    accuracy = correct / y_test.shape[0]

    print(f'Test Accuracy: {accuracy}%')
