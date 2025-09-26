import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# Config
# =========================
FRAMES_DIR = "./data/frames/UCSDped2/Test"  # folder video frames
RESULTS_DIR = "./results"
MODEL_PATH = "./convlstm_ae.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
IMG_SIZE = (64, 64)  # sesuai training
HIDDEN_DIM = 64

# =========================
# Dataset
# =========================
class VideoFrameDataset(Dataset):
    def __init__(self, folder_path, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")])

    def __len__(self):
        return max(0, len(self.frames) - self.seq_len + 1)

    def __getitem__(self, idx):
        seq = []
        for i in range(self.seq_len):
            img = cv2.imread(self.frames[idx + i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            seq.append(img)
        seq = np.array(seq, dtype=np.float32)
        seq = np.expand_dims(seq, axis=1)  # (seq_len, 1, H, W)
        return torch.tensor(seq)

# =========================
# ConvLSTM-AE Model
# =========================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_c):
        h, c = h_c
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.encoder_lstm = ConvLSTMCell(32, hidden_dim, 3)
        self.decoder_lstm = ConvLSTMCell(hidden_dim, 32, 3)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_dim, 3, padding=1),
            nn.Sigmoid()
        )
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, t, c, h, w = x.size()
        h_enc = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        c_enc = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        h_dec = torch.zeros(b, 32, h, w, device=x.device)
        c_dec = torch.zeros(b, 32, h, w, device=x.device)

        # Encoder
        for t_i in range(t):
            x_t = self.encoder_conv(x[:, t_i])
            h_enc, c_enc = self.encoder_lstm(x_t, (h_enc, c_enc))

        # Decoder
        outputs = []
        for t_i in range(t):
            h_dec, c_dec = self.decoder_lstm(h_enc, (h_dec, c_dec))
            out = self.decoder_conv(h_dec)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs

# =========================
# Utils
# =========================
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_heatmap(frame, recon, save_path):
    diff = np.abs(frame - recon)
    diff = (diff * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)

# =========================
# Main inference
# =========================
def main():
    model = ConvLSTMAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    make_folder(RESULTS_DIR)

    for root, dirs, files in os.walk(FRAMES_DIR):
        jpg_files = [f for f in files if f.endswith(".jpg")]
        if not jpg_files:
            continue

        rel_path = os.path.relpath(root, FRAMES_DIR)
        out_dir = os.path.join(RESULTS_DIR, rel_path)
        make_folder(out_dir)

        dataset = VideoFrameDataset(root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"[INFO] Processing {rel_path}, sequences: {len(dataset)}")
        for seq_idx, seq in enumerate(tqdm(loader)):
            seq = seq.to(DEVICE)
            with torch.no_grad():
                recon = model(seq)
            seq_np = seq.cpu().numpy()[0, :, 0]  # (seq_len, H, W)
            recon_np = recon.cpu().numpy()[0, :, 0]

            for i in range(SEQ_LEN):
                frame = (seq_np[i]*255).astype(np.uint8)
                recon_frame = (recon_np[i]*255).astype(np.uint8)
                save_path = os.path.join(out_dir, f"{seq_idx*SEQ_LEN + i + 1:05d}.jpg")
                save_heatmap(frame, recon_frame, save_path)

        print(f"[INFO] Saved results to {out_dir}")

if __name__ == "__main__":
    main()
