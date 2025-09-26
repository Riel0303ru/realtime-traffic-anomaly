import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# =========================
# CONFIG
# =========================
FRAMES_DIR = r"C:\VsCode\pedestrian_anomaly_detection\data\frames\UCSDped2\Test"
RESULTS_DIR = r"C:\VsCode\pedestrian_anomaly_detection\results"
MODEL_PATH = r"C:\VsCode\pedestrian_anomaly_detection\src\convlstm_ae.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
IMG_SIZE = (64, 64)

# =========================
# DATASET
# =========================
class VideoFrameDataset(Dataset):
    def __init__(self, video_folder, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.frames = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".jpg")])

    def __len__(self):
        return max(0, len(self.frames) - SEQ_LEN + 1)

    def __getitem__(self, idx):
        seq = []
        for i in range(self.seq_len):
            img = cv2.imread(self.frames[idx + i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            seq.append(img)
        seq = np.expand_dims(np.array(seq, dtype=np.float32), axis=1)  # (seq_len, 1, H, W)
        return torch.tensor(seq, dtype=torch.float32)

# =========================
# MODEL
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
    def __init__(self, input_dim=1, hidden_dim=64, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.encoder_lstm = ConvLSTMCell(32, hidden_dim, 3)
        self.decoder_lstm = ConvLSTMCell(hidden_dim, 32, 3)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, input_dim, 3, padding=1), nn.Sigmoid()
        )
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, t, C, H, W = x.size()
        h_enc = torch.zeros(b, self.hidden_dim, H, W, device=x.device)
        c_enc = torch.zeros(b, self.hidden_dim, H, W, device=x.device)
        h_dec = torch.zeros(b, 32, H, W, device=x.device)
        c_dec = torch.zeros(b, 32, H, W, device=x.device)

        # Encoder
        for i in range(t):
            x_t = self.encoder_conv(x[:, i])
            h_enc, c_enc = self.encoder_lstm(x_t, (h_enc, c_enc))

        # Decoder
        outputs = []
        for i in range(t):
            h_dec, c_dec = self.decoder_lstm(h_enc, (h_dec, c_dec))
            out = self.decoder_conv(h_dec)
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# =========================
# UTILS
# =========================
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_heatmap(frame, recon, save_path):
    # frame & recon: (1, H, W)
    frame = (frame*255).astype(np.uint8)
    recon = (recon*255).astype(np.uint8)
    diff = np.abs(frame - recon)  # (1, H, W)

    # ubah jadi 3 channel
    diff_3c = cv2.merge([diff[0], diff[0], diff[0]])  # (H, W, 3)
    frame_3c = cv2.merge([frame[0], frame[0], frame[0]])  # (H, W, 3)

    heatmap = cv2.applyColorMap(diff[0], cv2.COLORMAP_JET)  # atau diff_3c[:,:,0] juga bisa
    overlay = cv2.addWeighted(frame_3c, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


# =========================
# MAIN
# =========================
def main():
    make_folder(RESULTS_DIR)
    model = ConvLSTMAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Loop tiap video folder
    for video_folder in sorted(os.listdir(FRAMES_DIR)):
        video_path = os.path.join(FRAMES_DIR, video_folder)
        if not os.path.isdir(video_path):
            continue

        dataset = VideoFrameDataset(video_path)
        if len(dataset) == 0:
            print(f"[SKIP] Video folder {video_folder} kurang frame")
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        out_folder = os.path.join(RESULTS_DIR, video_folder)
        make_folder(out_folder)

        scores = []
        print(f"[INFO] Processing {video_folder} ...")
        for idx, seq in enumerate(tqdm(loader)):
            seq = seq.to(DEVICE)
            with torch.no_grad():
                recon = model(seq)
            seq_np = seq.cpu().numpy()
            recon_np = recon.cpu().numpy()
            for i in range(seq_np.shape[1]):
                score = np.mean((seq_np[0,i] - recon_np[0,i])**2)
                scores.append(score)
                save_path = os.path.join(out_folder, f"{idx*SEQ_LEN + i + 1:05d}.jpg")
                save_heatmap(seq_np[0,i], recon_np[0,i], save_path)

        # Save anomaly scores ke CSV
        df = pd.DataFrame({"frame": list(range(1,len(scores)+1)), "score": scores})
        df.to_csv(os.path.join(out_folder, "anomaly_scores.csv"), index=False)
        print(f"[INFO] Results saved to {out_folder}")

if __name__ == "__main__":
    main()
