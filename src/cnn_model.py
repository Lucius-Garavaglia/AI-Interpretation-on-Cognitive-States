import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_classifier
from data_loader import load_processed_data

class EEGFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EEGFeatureMLP(nn.Module):
    def __init__(self, num_features=67, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def train_model(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions)
    return np.array(y_true), np.array(y_pred)

def run_cnn_experiment(test_size=0.2, random_state=42, batch_size=128, epochs=30, lr=1e-3):
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = EEGFeatureDataset(X_train, y_train)
    test_dataset = EEGFeatureDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGFeatureMLP(num_features=X.shape[1]).to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{epochs} - loss: {loss:.4f}')

    y_true, y_pred = evaluate_model(model, test_loader, device)
    results = evaluate_classifier(y_true, y_pred, plot_confusion=True)
    return model, results

if __name__ == '__main__':
    run_cnn_experiment()