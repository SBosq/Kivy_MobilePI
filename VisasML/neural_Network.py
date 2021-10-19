"""import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from pylab import rcParams, plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch import nn, optim
import torch.nn.functional as F
from termcolor import colored as cl
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Heavily based on https://github.com/Prodicode/ann-visualizer

def ann_viz(model, view=True, filename="network.gv"):
    # Vizualizez a Sequential model.

    # Arguments
        # model: A Keras model instance.

        # view: whether to display the model after generation.

        # filename: where to save the vizualization. (a .gv file)

        # title: A title for the graph
    from graphviz import Digraph

    input_layer = 0
    hidden_layers_nr = 0
    layer_types = []
    hidden_layers = []
    output_layer = 0
    layers = [layer for layer in model.modules() if type(layer) == torch.nn.Linear]

    for layer in layers:
        if layer == layers[0]:
            input_layer = layer.in_features
            hidden_layers_nr += 1
            if type(layer) == torch.nn.Linear:
                hidden_layers.append(layer.out_features)
                layer_types.append("Dense")
            else:
                raise Exception("Input error")

        else:
            if layer == layers[-1]:
                output_layer = layer.out_features
            else:
                hidden_layers_nr += 1
                if type(layer) == torch.nn.Linear:

                    hidden_layers.append(layer.out_features)
                    layer_types.append("Dense")
                else:
                    raise Exception("Hidden error")
        last_layer_nodes = input_layer
        nodes_up = input_layer

    g = Digraph("g", filename=filename)
    n = 0
    g.graph_attr.update(splines="false", nodesep="0.5", ranksep="0", rankdir='LR')
    # Input Layer
    with g.subgraph(name="cluster_input") as c:
        if type(layers[0]) == torch.nn.Linear:
            the_label = "Input Layer"
            if layers[0].in_features > 10:
                the_label += " (+" + str(layers[0].in_features - 10) + ")"
                input_layer = 10
            c.attr(color="white")
            for i in range(0, input_layer):
                n += 1
                c.node(str(n))
                c.attr(labeljust="1")
                c.attr(label=the_label, labelloc="bottom")
                c.attr(rank="same")
                c.node_attr.update(
                    width="0.65",
                    style="filled",
                    shape="circle",
                    color=HAPPY_COLORS_PALETTE[3],
                    fontcolor=HAPPY_COLORS_PALETTE[3],
                )
    for i in range(0, hidden_layers_nr):
        with g.subgraph(name="cluster_" + str(i + 1)) as c:
            if layer_types[i] == "Dense":
                c.attr(color="white")
                c.attr(rank="same")
                the_label = f'Hidden Layer {i + 1}'
                if layers[i].out_features > 10:
                    the_label += " (+" + str(layers[i].out_features - 10) + ")"
                    hidden_layers[i] = 10
                c.attr(labeljust="right", labelloc="b", label=the_label)
                for j in range(0, hidden_layers[i]):
                    n += 1
                    c.node(
                        str(n),
                        width="0.65",
                        shape="circle",
                        style="filled",
                        color=HAPPY_COLORS_PALETTE[0],
                        fontcolor=HAPPY_COLORS_PALETTE[0],
                    )
                    for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                        g.edge(str(h), str(n))
                last_layer_nodes = hidden_layers[i]
                nodes_up += hidden_layers[i]
            else:
                raise Exception("Hidden layer type not supported")

    with g.subgraph(name="cluster_output") as c:
        if type(layers[-1]) == torch.nn.Linear:
            c.attr(color="white")
            c.attr(rank="same")
            c.attr(labeljust="1")
            for i in range(1, output_layer + 1):
                n += 1
                c.node(
                    str(n),
                    width="0.65",
                    shape="circle",
                    style="filled",
                    color=HAPPY_COLORS_PALETTE[4],
                    fontcolor=HAPPY_COLORS_PALETTE[4],

                )
                for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                    g.edge(str(h), str(n))
            c.attr(label="Output Layer", labelloc="bottom")
            c.node_attr.update(
                color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle"
            )

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070", penwidth="2")
    if view is True:
        g.view()

    return g


sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

visas1 = pd.read_csv('unified_Visas.csv')
sns.countplot(visas1.CASE_STATUS)
# plt.show()
for (columnName, columnData) in visas1.iteritems():
    visas1[columnName] = visas1[columnName].astype('category')
    visas1[columnName] = visas1[columnName].cat.codes
scaler = MinMaxScaler()
normVisas = pd.DataFrame(scaler.fit_transform(visas1), columns=visas1.columns, index=visas1.index)
X_var = normVisas[['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                   'PREVAILING_WAGE_1']]
y_var = normVisas['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3)
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())


class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


net = Net(X_train.shape[1])
# ann_viz(net)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)

criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


for epoch in range(1000):

    y_pred = net(X_train)

    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)

    if epoch % 100 == 0:
        train_acc = calculate_accuracy(y_train, y_pred)

        y_test_pred = net(X_test)
        y_test_pred = torch.squeeze(y_test_pred)

        test_loss = criterion(y_test_pred, y_test)

        test_acc = calculate_accuracy(y_test, y_test_pred)
        print(
            f'''epoch {epoch}
Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()


classes = ['CERTIFIED', 'DENIED']

y_pred = net(X_test)

y_pred = y_pred.ge(.5).view(-1).cpu()
y_test = y_test.cpu()

print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']),
      '\n')
print(classification_report(y_test, y_pred, target_names=classes))
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from termcolor import colored as cl


visas1 = pd.read_csv('normVisas.csv')
X_var = visas1[['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                'PREVAILING_WAGE_1']].values
y_var = visas1['CASE_STATUS'].values
sc = StandardScaler()
X_var = sc.fit_transform(X_var)
ohe = OneHotEncoder()
y_var = y_var.reshape(-1, 1)
y_var = ohe.fit_transform(y_var).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3)
model = Sequential()
model.add(Dense(4, input_dim=6, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=64)
# model.save("visas_NN.h5")
y_pred = model.predict(X_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
a = accuracy_score(pred, test)
print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(a * 100, 2), attrs=['bold']), '\n')
reconstructed_model = keras.models.load_model("visas_NN.h5")
# np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))
reconstructed_model.fit(X_train, y_train, epochs=100, batch_size=64)
pred1 = list()
for i in range(len(y_pred)):
    pred1.append(np.argmax(y_pred[i]))
test1 = list()
for i in range(len(y_test)):
    test1.append(np.argmax(y_test[i]))
a1 = accuracy_score(pred1, test1)
print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(a1 * 100, 2), attrs=['bold']), '\n')
