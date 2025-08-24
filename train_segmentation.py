# train_segmentation.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# --- Simple U-Net-like stub ---
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        return self.dec(self.enc(x))

def train_segmentation():
    print("Training dummy segmentation model...")
    model = TinyUNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Fake training loop
    for epoch in range(3):
        x = torch.rand(2, 3, 64, 64)   # fake input
        y = torch.rand(2, 1, 64, 64)   # fake mask
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    # Save checkpoint
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/unet.pt")
    print("✅ Saved segmentation model to artifacts/unet.pt")

if __name__ == "__main__":
    train_segmentation()
