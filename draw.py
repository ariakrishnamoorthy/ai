import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import pickle
import os
import cv2
from scipy import ndimage

from ann import Layer, Network3, Neuron

SAVE_PATH = 'network.pkl'

if not os.path.exists(SAVE_PATH):
    print(f"Error: {SAVE_PATH} not found. Please run your training script first.")
    exit()

print("Loading saved network...")
with open(SAVE_PATH, 'rb') as f:
    network = pickle.load(f)
print("Network loaded successfully!")


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit (0-9)")

        self.canvas_width = 420
        self.canvas_height = 420

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(pady=15)

        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)

        # ── NEW: result label ──────────────────────────────────────────────
        self.result_var = tk.StringVar(value="Draw a digit, then click Predict")
        self.result_label = tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Arial", 22, "bold"),
            fg="#1a73e8",
            pady=8,
        )
        self.result_label.pack()
        # ──────────────────────────────────────────────────────────────────

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=15)

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict, font=("Arial", 16))
        predict_btn.pack(side=tk.LEFT, padx=40)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear, font=("Arial", 16))
        clear_btn.pack(side=tk.RIGHT, padx=40)

    def paint(self, event):
        brush_size = 18
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)

        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='black')
        # ── NEW: reset label on clear ──────────────────────────────────────
        self.result_var.set("Draw a digit, then click Predict")
        self.result_label.config(fg="#1a73e8")
        # ──────────────────────────────────────────────────────────────────

    def predict(self):
        img_array = np.array(self.image)

        coords = cv2.findNonZero(img_array)
        if coords is None:
            # ── NEW: show empty-canvas warning in the label too ────────────
            self.result_var.set("⚠️  Canvas is empty!")
            self.result_label.config(fg="#e85c1a")
            # ──────────────────────────────────────────────────────────────
            print("Canvas is empty!")
            return

        x, y, w, h = cv2.boundingRect(coords)
        cropped_digit = img_array[y:y+h, x:x+w]

        if h > w:
            new_h = 20
            new_w = max(1, int(20 * w / h))
        else:
            new_w = 20
            new_h = max(1, int(20 * h / w))

        resized_digit = cv2.resize(cropped_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cy, cx = ndimage.center_of_mass(resized_digit)

        final_image = np.zeros((28, 28), dtype=np.float32)
        start_y = int(round(14.0 - cy))
        start_x = int(round(14.0 - cx))
        final_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

        flattened_image = final_image.flatten() / 255.0

        network.inputLayer.changeNeuronActivations(flattened_image)
        network.forwardPass()
        network.softmax()

        predicted_digit = np.argmax(network.softMaxOutput)
        confidence = network.softMaxOutput[predicted_digit] * 100
        print(f"Network Prediction: {predicted_digit}  (Confidence: {confidence:.2f}%)")

        # ── NEW: update the on-screen label ───────────────────────────────
        self.result_var.set(f"Predicted: {predicted_digit}   ({confidence:.1f}% confident)")
        self.result_label.config(fg="#1a8a2e" if confidence >= 80 else "#e85c1a")
        # ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()