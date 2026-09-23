# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import time
import zipfile
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

# -------- CONFIG ----------
MODEL_PATH = r"D:\Github\brain-segmentation\best_model.pth"  # chỉnh nếu cần
INPUT_SIZE = 256  
BATCH_SIZE = 8
NUM_CLASSES = 4
CLASS_NAMES = ["BG","Viable Tumor","Fibrosis/Hyalinization","Non-tumor"]
# màu (BGR để dùng với OpenCV) — bạn có thể đổi
CLASS_COLORS = {
    0: (0,0,0),         # BG - black
    1: (0,0,255),       # VT - red
    2: (0,255,255),     # FH - yellow
    3: (0,255,0)        # NT - green
}
# ---------------------------

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load your model architecture first (replace with your class)
    # Example: model = MyUNet(num_classes=NUM_CLASSES)
    # For demo, we assume saved state dict contains full model (torch.save(model))
    try:
        model = torch.load(path, map_location=device)
    except Exception as e:
        st.error(f"Không load được model tự động: {e}")
        raise
    model.to(device)
    model.eval()
    return model, device

def preprocess_patch(pil_img, size=INPUT_SIZE):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # normalize? (tùy model) — nếu training có normalization thì áp lại
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # 1,C,H,W
    return tensor

def sliding_window_inference(img_pil, model, device, patch_size=INPUT_SIZE, stride=None):
    if stride is None:
        stride = patch_size // 2
    img = np.array(img_pil.convert("RGB"))
    h,w,_ = img.shape
    # pad to cover edges
    pad_h = ( ( (h - patch_size) // stride + 1) * stride + patch_size ) - h if h > patch_size else patch_size - h
    pad_w = ( ( (w - patch_size) // stride + 1) * stride + patch_size ) - w if w > patch_size else patch_size - w
    img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    H_pad, W_pad = img_pad.shape[:2]
    count_map = np.zeros((NUM_CLASSES, H_pad, W_pad), dtype=np.float32)
    score_map = np.zeros((NUM_CLASSES, H_pad, W_pad), dtype=np.float32)

    patches = []
    coords = []
    for y in range(0, H_pad - patch_size + 1, stride):
        for x in range(0, W_pad - patch_size + 1, stride):
            patch = img_pad[y:y+patch_size, x:x+patch_size]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = patch.astype(np.float32) / 255.0
            patches.append(patch)
            coords.append((x,y))
    # batch predict
    transform = T.Compose([T.ToTensor()])  # if needed
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch = patches[i:i+BATCH_SIZE]
            batch_t = torch.stack([torch.from_numpy(b).permute(2,0,1) for b in batch]).to(device).float()
            # if model needs normalization, apply here
            logits = model(batch_t)  # expect [N, C, H, W]
            if logits.ndim == 3:
                logits = logits.unsqueeze(0)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # [N,C,H,W]
            all_preds.append(probs)
    all_preds = np.concatenate(all_preds, axis=0)
    # add to score_map
    for (x,y), probs in zip(coords, all_preds):
        score_map[:, y:y+patch_size, x:x+patch_size] += probs
        count_map[:, y:y+patch_size, x:x+patch_size] += 1.0
    # avoid div by zero
    avg_score = score_map / np.maximum(count_map, 1e-6)
    mask = np.argmax(avg_score, axis=0).astype(np.uint8)  # H_pad x W_pad
    # crop to original
    mask = mask[:h, :w]
    return mask

def mask_to_overlay(img_pil, mask, alpha=0.5):
    img = np.array(img_pil.convert("RGB"))
    overlay = img.copy()
    for cls, col in CLASS_COLORS.items():
        b,g,r = col
        color = np.array([r,g,b], dtype=np.uint8)  # we use RGB ordering in img
        overlay[mask==cls] = (overlay[mask==cls] * (1-alpha) + color * alpha).astype(np.uint8)
    return overlay

def compute_percentages(mask):
    total = mask.size
    results = {}
    for cls_idx, name in enumerate(CLASS_NAMES):
        area = (mask==cls_idx).sum()
        pct = 100.0 * area / total
        results[name] = {"area": int(area), "percent": float(pct)}
    return results

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Histopath Seg Demo")
st.title("Histopathology segmentation — Demo app")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload WSI/PNG", type=["png","jpg","jpeg","tif","tiff"])
    use_demo = st.checkbox("Use demo sample", value=True)
    if use_demo:
        demo_path = None  # nếu có demo image, load path thay vì upload
        st.write("Using demo image (you can uncheck to upload your own).")
    patch_size = st.number_input("Patch size", value=INPUT_SIZE, step=64)
    run_btn = st.button("Run inference")

with col2:
    st.header("Output")
    status = st.empty()

# load model
model = None
device = None
try:
    model, device = load_model(MODEL_PATH)
    st.sidebar.success(f"Model loaded. Device: {device}")
except Exception as e:
    st.sidebar.error("Failed to load model — kiểm tra đường dẫn/model.")
    model = None
    device = None

img_pil = None
if uploaded is not None:
    img_pil = Image.open(uploaded)
elif use_demo:
    # nếu bạn có hình demo, load ở đây thay path
    # fallback: tạo ảnh ngẫu nhiên cho demo
    img_pil = Image.new("RGB", (1024,1024), color=(255,255,255))
    cv2.putText(np.array(img_pil), "Demo Image", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 5)
else:
    st.info("Upload file hoặc bật 'Use demo sample' để thử.")

if img_pil is not None:
    st.image(img_pil, caption="Input image (preview)", use_container_width=True)

if run_btn and img_pil is not None:
    if model is None:
        st.error("Model chưa được load. Kiểm tra `MODEL_PATH` hoặc khởi động lại server.")
    else:
        t0 = time.time()
        status.info("Running inference (this may take a while for large images)...")
        mask = sliding_window_inference(img_pil, model, device, patch_size=patch_size, stride=patch_size//2)
        overlay = mask_to_overlay(img_pil, mask, alpha=0.5)
        perc = compute_percentages(mask)
        st.subheader("Segmentation result")
        st.image(overlay, caption="Overlay", use_container_width=True)
        st.write("Class percentages:")
        st.table({k: f"{v['percent']:.2f}% ({v['area']} px)" for k,v in perc.items()}.items())
        # download mask as png
        mask_img = Image.fromarray((mask.astype(np.uint8)))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download mask PNG", data=buf, file_name="mask.png", mime="image/png")
        # csv report
        import csv
        out_buf = io.StringIO()
        writer = csv.writer(out_buf)
        writer.writerow(["class","area_px","percent"])
        for k,v in perc.items():
            writer.writerow([k, v["area"], f"{v['percent']:.6f}"])
        st.download_button("Download report CSV", data=out_buf.getvalue(), file_name="report.csv", mime="text/csv")
        t1 = time.time()
        status.success(f"Done in {t1-t0:.1f}s")

st.markdown("---")
st.markdown("Notes: 1) Điều chỉnh `INPUT_SIZE` / normalization theo model thực tế; 2) Nếu WSI quá lớn, sử dụng tile server hoặc s3 streaming.")
