import torch
import torch.nn as nn
import pandas as pd
df = pd.read_csv("small_sample.csv")
feature_cols = ["TransactionAmt", "card1", "card3", "C1", "C2",
                "C3", "C4", "C5", "C13", "C14"]
X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y = torch.tensor(df["isFraud"].values, dtype=torch.float32)
mean = X.mean(dim=0)
std = X.std(dim=0)
X_normalized = (X - mean) / std
X_train = X_normalized[0:4000]
y_train = y[0:4000]
X_test = X_normalized[4000:5000]
y_test = y[4000:5000]
y_train = y_train.unsqueeze(dim=1)
y_test = y_test.unsqueeze(dim=1)
class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    # No sigmoid here -- BCEWithLogitsLoss applies sigmoid internally.
model = FraudNet()
fraud_count = y_train.sum()
non_fraud_count = len(y_train) - fraud_count
pos_weight = torch.tensor(non_fraud_count / fraud_count, dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
for epoch in range(epochs):
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")
# Testing on unseen data
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).float()
    accuracy = (test_preds == y_test).float().mean()
    print(f"Test accuracy: {accuracy.item():.4f}")
    print(f"Predicted fraud count: {test_preds.sum().item():.0f} / "
          f"actual fraud count: {y_test.sum().item():.0f}")
