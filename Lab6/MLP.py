import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_dataset, val_dataset = random_split(train_data, [int(0.8 * len(train_data)), len(train_data) - int(0.8 * len(train_data))])


class MLP(nn.Module):
    def __init__(self, h1, h2, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h2, 10)
        )
    def forward(self, x): return self.net(x)


def evaluate_params(params):
    h1, h2, lr, batch_size, dropout, l2 = params
    model = MLP(int(h1), int(h2), dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size))
    

    for _ in range(3):
        model.train()
        for img, lbl in train_loader:
            optimizer.zero_grad()
            criterion(model(img.to(device)), lbl.to(device)).backward()
            optimizer.step()
            
    model.eval()
    correct = sum((model(img.to(device)).argmax(1) == lbl.to(device)).sum().item() for img, lbl in val_loader)
    return -(100 * correct / len(val_dataset))

space = [
    Integer(256, 1024), Integer(128, 512), Real(1e-4, 1e-1, prior='log-uniform'),
    Integer(32, 256), Real(0.1, 0.5), Real(1e-6, 1e-2, prior='log-uniform')
]

print("Запуск байесовской оптимизации")
res = gp_minimize(evaluate_params, space, n_calls=10, n_random_starts=5, random_state=33, verbose=True)
bp = res.x


final_model = MLP(int(bp[0]), int(bp[1]), bp[4]).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=bp[2], weight_decay=bp[5])
criterion = nn.CrossEntropyLoss()
loaders = {
    'train': DataLoader(train_dataset, batch_size=int(bp[3]), shuffle=True),
    'val': DataLoader(val_dataset, batch_size=int(bp[3])),
    'test': DataLoader(test_data, batch_size=int(bp[3]))
}

def train_final(epochs=20):
    hist = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    best_acc, best_w = 0, None
    
    for e in range(epochs):
        final_model.train()
        t_loss, t_corr = 0, 0
        for img, lbl in loaders['train']:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = final_model(img)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_corr += (out.argmax(1) == lbl).sum().item()
            
        final_model.eval()
        v_loss, v_corr = 0, 0
        with torch.no_grad():
            for img, lbl in loaders['val']:
                img, lbl = img.to(device), lbl.to(device)
                out = final_model(img)
                v_loss += criterion(out, lbl).item()
                v_corr += (out.argmax(1) == lbl).sum().item()
        
        acc = 100 * v_corr / len(val_dataset)
        if acc > best_acc: best_acc, best_w = acc, final_model.state_dict()
        
        hist['t_loss'].append(t_loss / len(loaders['train']))
        hist['v_loss'].append(v_loss / len(loaders['val']))
        hist['t_acc'].append(100 * t_corr / len(train_dataset))
        hist['v_acc'].append(acc)
        print(f"Epoch {e+1}: Val Acc: {acc:.2f}%")

    if best_w: final_model.load_state_dict(best_w)
    return hist

history = train_final()


final_model.eval()
test_correct = 0
with torch.no_grad():
    for img, lbl in loaders['test']:
        img, lbl = img.to(device), lbl.to(device)
        test_correct += (final_model(img).argmax(1) == lbl).sum().item()
test_acc = 100 * test_correct / len(test_data)
print(f"\nTest Acc: {test_acc:.2f}%")


print(f"Max Train: {max(history['t_acc']):.2f}% | Max Val: {max(history['v_acc']):.2f}%")
print(f"Overfitting gap: {max(history['t_acc']) - max(history['v_acc']):.2f}%")


def plot_results():
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], hspace=0.3)
    
    ax_txt = fig.add_subplot(gs[0])
    ax_txt.axis('off')
    txt = (f"BEST MLP PARAMS:\nHidden1: {int(bp[0])} | Hidden2: {int(bp[1])}\n"
           f"LR: {bp[2]:.6f} | Batch: {int(bp[3])} | Drop: {bp[4]:.3f} | L2: {bp[5]:.6f}\n"
           f"Results: Val(max)={max(history['v_acc']):.1f}% | Test={test_acc:.2f}%")
    ax_txt.text(0.5, 0.5, txt, ha='center', va='center', fontfamily='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    gs2 = gs[1].subgridspec(1, 2, wspace=0.3)
    ax1, ax2 = fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1])
    
    ax1.plot(history['t_loss'], label='Train')
    ax1.plot(history['v_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2.plot(history['t_acc'], label=f"Train Max: {max(history['t_acc']):.1f}%")
    ax2.plot(history['v_acc'], label=f"Val Max: {max(history['v_acc']):.1f}%")
    ax2.axhline(test_acc, color='g', linestyle='--', label=f'Test: {test_acc:.1f}%')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(f'MLP Optimized (Test: {test_acc:.2f}%)', fontsize=14, fontweight='bold')
    plt.show()

plot_results()