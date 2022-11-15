import numpy as np
import torch
from torch import nn
import seaborn as sns
from torch_geometric.datasets import QM9
from torch_geometric.nn.conv.cg_conv import CGConv
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import global_add_pool as gap
import matplotlib.pyplot as plt
import covalent as ct
import torch.nn.functional as F

sns.set_style("ticks")
import os
cwd = os.getcwd()
print(cwd)
class GraphConvNet(nn.Module):
    def __init__(self):
        super(GraphConvNet, self).__init__()
        num_atomic_features=11 
        num_bond_features=4
        self.graph_conv_layer_1 = CGConv(channels=num_atomic_features, dim=num_bond_features, batch_norm=True)
        self.graph_conv_layer_2 = CGConv(channels=num_atomic_features, dim=num_bond_features, batch_norm=True)
        self.fc_layer_1 = nn.Linear(num_atomic_features, 5)
        self.fc_layer_2 = nn.Linear(5, 1)
        self.softplus = nn.Softplus()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # First Graph Convolution
        x = self.graph_conv_layer_1(x, edge_index, edge_attr)
        # Second Graph Convolution
        x = self.graph_conv_layer_2(x, edge_index, edge_attr)
        x = gap(x, batch)
        x = self.softplus(self.fc_layer_1(x))
        x = self.fc_layer_2(x)
        return x


@ct.electron(executor="local")
def train_single_epoch(train_loader, optimizer, model, loss, property_idx, epoch, device="cpu"):
    batch_loss_list = []
    model.train()
    for batch_data in train_loader:
        output = model(batch_data.to(device))
        target = batch_data.y[:, property_idx].unsqueeze(1).to(device)
        batch_loss = loss(output, target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_loss_list.append(batch_loss.item())
    return model, optimizer, np.mean(batch_loss_list)

@ct.electron(executor="local")
def validation(val_loader, model, loss, property_idx):
    batch_loss_list = []
    model.eval()
    for batch_data in val_loader:
        output = model(batch_data)
        target = batch_data.y[:, property_idx].unsqueeze(1)
        batch_loss = loss(output, target)
        batch_loss_list.append(batch_loss.item())
    return np.mean(batch_loss_list)

@ct.electron(executor="local")
def test(test_loader, model, loss, property_idx):
    model.eval()
    true_values = []
    predictions = []
    batch_loss_list = []
    for batch_data in test_loader:
        output = model(batch_data)
        target = batch_data.y[:, property_idx].unsqueeze(1)
        # Save predicted and true values for scatter plot
        true_values.append(target.detach().cpu().numpy())
        predictions.append(output.detach().cpu().numpy())
        batch_loss = loss(output, target)
        batch_loss_list.append(batch_loss.item())

    true_values = np.concatenate(true_values)
    predictions = np.concatenate(predictions)
    return np.mean(batch_loss_list), true_values, predictions

@ct.electron(executor="local")
def train_model(train_loader, val_loader, model, loss, n_epochs, property_idx):
    train_loss_list = []
    val_loss_list = []
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        model, optimizer, train_loss = train_single_epoch(train_loader, optimizer, model, loss, property_idx, epoch)
        val_loss = validation(val_loader, model, loss, property_idx)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f"Epoch: {epoch+1}, Trining Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
    return model, optimizer, train_loss_list, val_loss_list


def workflow(model, n_epochs = 20, train_size = 80, property_idx = 4):
    loss = F.mse_loss
    dataset = QM9(cwd)
    print("Len dataset: ", len(dataset))
    n_train = int(train_size*len(dataset)/100)
    n_val = int((100-train_size)*len(dataset)/200)
    n_test = len(dataset) - n_train - n_val
    train_set, validation_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
# #    # print(dataset[0].edge_attr.shape)
#     num_atomic_features = dataset[0].x.shape[1]
#     num_bond_features = dataset[0].edge_attr.shape[1]
#     model = net(num_atomic_features, num_bond_features)
    model, optimizer, train_loss_list, val_loss_list = train_model(train_loader, val_loader, model, loss, n_epochs, property_idx)
    test_loss, true_values, predictions = test(test_loader, model, loss, property_idx)
    return train_loss_list, val_loss_list, test_loss, true_values, predictions


n_epochs = 20
workflow = ct.lattice(workflow, executor="local")

dispatch_id = ct.dispatch(workflow)(model=GraphConvNet().to("cpu"), n_epochs=n_epochs)
print(f"Dispatch id: {dispatch_id}")
results = ct.get_result(dispatch_id=dispatch_id, wait=True)
print(f"Covalent workflow takes {results.end_time - results.start_time} seconds.")
#print(results.result)
train_loss_list, val_loss_list, test_loss, true_values, predictions = results.result
#print(train_loss_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# Plot training curves 
ax1.plot(range(1, n_epochs+1), train_loss_list, label="Traing Loss")
ax1.plot(range(1, n_epochs+1), val_loss_list, label="Validation Loss")
ax1.set_xticks(np.arange(1, n_epochs+2, 2))
ax1.set_title("Training curves")
ax1.set_ylabel("MSE Loss, eV$^2$")
ax1.set_xlabel("Epochs")
ax1.legend()
# Plot Predictions vs True valuses
ax2.scatter(true_values, predictions, alpha=0.7, label=f"Test set MSE: {test_loss:.3f} eV$^2$")
ax2.axline((1, 1), slope=1)
ax2.set_title("Test set predictions")
ax2.set_xlabel("HOMO-LUMO gap, eV")
ax2.set_ylabel("Predicted HOMO-LUMO gap, eV")
ax2.set_xlim([1, 11])
ax2.set_ylim([1, 11])
ax2.legend()
fig.savefig('results.png', dpi=fig.dpi)