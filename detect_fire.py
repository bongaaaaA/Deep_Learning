import cv2
import torch
from torchvision import transforms
from PIL import Image
from fire_classifier import model

# ===================== GPU setup ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("fire_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# ===================== Transform ======================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

window_size = 64
stride = 64  

# ==================== Video Capture ==================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    scale_factor = 0.5
    frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    windows = []
    coords = []

    # ================= Sliding Window ====================
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = Image.fromarray(img_rgb[y:y + window_size, x:x + window_size])
            windows.append(transform(window))
            coords.append((x, y, x + window_size, y + window_size))

    if windows:
        input_batch = torch.stack(windows).to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            probs = torch.softmax(outputs, dim=1)[:, 0]  # fire probs

        fire_boxes = [coords[i] for i, p in enumerate(probs) if p > 0.9]
    else:
        fire_boxes = []

    # Decision
    if fire_boxes:
        # Merge all boxes into ONE
        x1 = min(b[0] for b in fire_boxes)
        y1 = min(b[1] for b in fire_boxes)
        x2 = max(b[2] for b in fire_boxes)
        y2 = max(b[3] for b in fire_boxes)

        x1 = int(x1 / scale_factor)
        y1 = int(y1 / scale_factor)
        x2 = int(x2 / scale_factor)
        y2 = int(y2 / scale_factor)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = "Non Fire"
        color = (0, 255, 0)
    else:
        label = "Fire"
        color = (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ===============================  Done ============================