# train_captioning.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# --- Simple CNN+LSTM captioning stub (fixed) ---
class TinyCaptionModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # (B, 8, 16, 16)
            nn.ReLU(),
            nn.Flatten()  # (B, 2048)
        )
        self.fc_img = nn.Linear(2048, hidden_dim)   # project to hidden
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        feats = self.cnn(x)              # (B, 2048)
        feats = self.fc_img(feats)       # (B, hidden_dim)
        feats = feats.unsqueeze(1)       # (B, 1, hidden_dim)
        out, _ = self.lstm(feats)        # (B, 1, hidden_dim)
        return self.fc_out(out.squeeze(1))  # (B, vocab_size)

def train_captioning():
    print("Training dummy captioning model...")
    model = TinyCaptionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Fake training loop
    for epoch in range(3):
        x = torch.rand(2, 3, 32, 32)          # fake image batch
        y = torch.randint(0, 50, (2,))        # fake labels (2 samples)
        pred = model(x)                       # (2, vocab_size)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    # Save checkpoint
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/caption_cnn_lstm.pt")
    print("✅ Saved captioning model to artifacts/caption_cnn_lstm.pt")

if __name__ == "__main__":
    train_captioning()
