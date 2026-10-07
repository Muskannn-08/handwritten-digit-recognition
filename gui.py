"""
gui.py  (version 5)
- Write ANY number of digits (3, 4, 6, up to 8 on this canvas)
- Mode 1: Boxes (most accurate): one digit per box, empty boxes are ignored
- Mode 2: Free drawing: digits are split automatically
- Warns when the model is unsure
- "Teach" box: type the correct number -> saved as training samples of YOUR
  handwriting -> run finetune.py to boost accuracy

Run:  python gui.py
"""
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

from predict import load_models, recognize

MAX_SLOTS = 8        # number of boxes (increase for longer numbers)
SLOT_W = 110
CANVAS_W = SLOT_W * MAX_SLOTS
CANVAS_H = 250
BRUSH = 14           # pen width
LOW_CONF = 70.0      # below this % we show a warning
USER_FILE = os.path.join("user_data", "samples.npz")


def sample_count():
    if os.path.exists(USER_FILE):
        return len(np.load(USER_FILE)["y"])
    return 0


class DigitApp:
    def __init__(self, root):
        self.root = root
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        self.models = load_models()
        self.last = None
        self.last_digits = []
        self.tk_previews = []

        tk.Label(root, text="HANDWRITTEN NUMBER RECOGNIZER",
                 font=("Arial", 15, "bold")).pack(pady=(8, 2))
        self.hint = tk.Label(root, font=("Arial", 9), fg="gray30")
        self.hint.pack()

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                                bg="black", cursor="cross")
        self.canvas.pack(padx=15, pady=6)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        # hidden PIL image that mirrors the canvas
        self.image = Image.new("L", (CANVAS_W, CANVAS_H), 0)
        self.draw = ImageDraw.Draw(self.image)

        # mode selection
        self.mode = tk.StringVar(value="boxes")
        mode_row = tk.Frame(root)
        mode_row.pack()
        tk.Radiobutton(mode_row, text="Boxes (most accurate)", variable=self.mode,
                       value="boxes", command=self.toggle_guides).pack(side="left", padx=8)
        tk.Radiobutton(mode_row, text="Free drawing (auto-split)", variable=self.mode,
                       value="auto", command=self.toggle_guides).pack(side="left", padx=8)
        self.toggle_guides()

        bar = tk.Frame(root)
        bar.pack(pady=6)
        tk.Button(bar, text="Clear", width=10, command=self.clear).grid(row=0, column=0, padx=4)
        tk.Button(bar, text="Recognize", width=10, command=self.recognize).grid(row=0, column=1, padx=4)
        tk.Button(bar, text="Upload", width=10, command=self.upload).grid(row=0, column=2, padx=4)

        self.result = tk.Label(root, text="Number: -", font=("Arial", 24, "bold"))
        self.result.pack()
        self.detail = tk.Label(root, text="", font=("Arial", 10), fg="gray30", justify="left")
        self.detail.pack()

        tk.Label(root, text="What the model sees (28x28 per digit):",
                 font=("Arial", 9)).pack(pady=(6, 0))
        self.preview_frame = tk.Frame(root)
        self.preview_frame.pack(pady=4)

        # teach section
        tk.Frame(root, height=1, bg="gray70").pack(fill="x", padx=15, pady=6)
        tk.Label(root, text="Prediction wrong? Type the CORRECT number and save it "
                            "(improves accuracy on your handwriting):",
                 font=("Arial", 9)).pack()
        teach = tk.Frame(root)
        teach.pack(pady=4)
        self.entry = tk.Entry(teach, width=14, font=("Arial", 12))
        self.entry.pack(side="left", padx=4)
        tk.Button(teach, text="Save as training sample",
                  command=self.save_sample).pack(side="left", padx=4)
        self.count_label = tk.Label(root, font=("Arial", 9), fg="gray30")
        self.count_label.pack(pady=(0, 10))
        self.update_count()

    # ---------- guide boxes ----------
    def toggle_guides(self):
        self.canvas.delete("guide")
        if self.mode.get() == "boxes":
            for i in range(1, MAX_SLOTS):
                x = i * SLOT_W
                self.canvas.create_line(x, 0, x, CANVAS_H, fill="#555555",
                                        dash=(4, 4), tags="guide")
            self.canvas.tag_lower("guide")
            self.hint.config(text="Write ONE big digit per box, starting from the left. "
                                  "Stay inside the lines. Unused boxes are ignored.")
        else:
            self.hint.config(text="Write big digits with a clear gap between them (digits must not touch). Use Boxes mode if unsure.")

    # ---------- drawing ----------
    def start_stroke(self, e):
        self.last = (e.x, e.y)
        self.dot(e.x, e.y)

    def dot(self, x, y):
        r = BRUSH / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white", tags="ink")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def paint(self, e):
        if self.last is None:
            self.last = (e.x, e.y)
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=BRUSH,
                                capstyle=tk.ROUND, tags="ink")
        self.draw.line([x0, y0, e.x, e.y], fill=255, width=BRUSH)
        self.dot(e.x, e.y)
        self.last = (e.x, e.y)

    def end_stroke(self, e):
        self.last = None

    def clear(self):
        self.canvas.delete("ink")
        self.draw.rectangle([0, 0, CANVAS_W, CANVAS_H], fill=0)
        self.result.config(text="Number: -")
        self.detail.config(text="")
        self.last_digits = []
        for w in self.preview_frame.winfo_children():
            w.destroy()

    # ---------- recognition ----------
    def recognize(self):
        self.run(self.image, auto_invert=False, mode=self.mode.get())

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.run(Image.open(path), auto_invert=True, mode="auto")

    def run(self, img, auto_invert, mode):
        number, details, digit_imgs = recognize(self.models, img, auto_invert=auto_invert,
                                                mode=mode, n_slots=MAX_SLOTS)
        if not digit_imgs:
            messagebox.showinfo("Nothing found", "Please draw a digit first.")
            return
        self.last_digits = digit_imgs
        self.result.config(text=f"Number: {number}")

        lines = []
        for i, (d, c, top3) in enumerate(details, 1):
            alt = ", ".join(f"{a} ({b:.0f}%)" for a, b in top3)
            warn = "   <-- not sure, check this one" if c < LOW_CONF else ""
            lines.append(f"Digit {i}: {d}  |  {c:.1f}%  |  top 3: {alt}{warn}")
        self.detail.config(text="\n".join(lines))

        for w in self.preview_frame.winfo_children():
            w.destroy()
        self.tk_previews = []
        for arr, (_, c, _) in zip(digit_imgs, details):
            im = Image.fromarray((arr * 255).astype("uint8")).resize((70, 70), Image.NEAREST)
            tkimg = ImageTk.PhotoImage(im)
            self.tk_previews.append(tkimg)
            tk.Label(self.preview_frame, image=tkimg, bg="black", borderwidth=0,
                     highlightthickness=3,
                     highlightbackground="red" if c < LOW_CONF else "gray40"
                     ).pack(side="left", padx=3)

    # ---------- teach: save samples of your own handwriting ----------
    def save_sample(self):
        text = self.entry.get().strip()
        if not self.last_digits:
            messagebox.showinfo("Nothing to save", "Draw a number and click Recognize first.")
            return
        if not text.isdigit() or len(text) != len(self.last_digits):
            messagebox.showwarning(
                "Check the number",
                f"The model found {len(self.last_digits)} digit(s), so type exactly "
                f"{len(self.last_digits)} digit(s).\n"
                "If it found the wrong number of digits, redraw with clearer gaps/boxes.")
            return

        x_new = np.array(self.last_digits, dtype="float32")
        y_new = np.array([int(c) for c in text], dtype="int64")
        os.makedirs("user_data", exist_ok=True)
        if os.path.exists(USER_FILE):
            old = np.load(USER_FILE)
            x_new = np.concatenate([old["x"], x_new])
            y_new = np.concatenate([old["y"], y_new])
        np.savez_compressed(USER_FILE, x=x_new, y=y_new)

        self.entry.delete(0, "end")
        self.update_count()

    def update_count(self):
        self.count_label.config(
            text=f"Saved samples: {sample_count()}   "
                 f"(aim for 20+ per digit, then run:  python finetune.py)")


if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()