import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

# ============================================================
# 1. DATA + MODEL (unchanged logic)
# ============================================================
dataset = pd.read_csv(r"E:\ds&ml\student_placement_dataset.csv")

FEATURE_COLUMNS = ["CGPA", "Backlogs", "Communication", "Projects", "Internships", "Technical_Skills"]
TARGET_COLUMN = "Placement_Status"

x = dataset[FEATURE_COLUMNS].values
y = np.asarray(dataset[TARGET_COLUMN], dtype=object)   # plain NumPy array (avoids Arrow-array error)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("Confusion matrix:\n", cm)
acc = (sum(np.diag(cm)) / len(y_test_encoded))
print("Model accuracy:", acc)

# ============================================================
# 2. COLORS
# ============================================================
BG            = "#e4e9fb"   # window background
CARD_BG       = "#ffffff"
NAVY          = "#1e3a8a"
BLUE          = "#2563eb"
BLUE_HOVER    = "#1d4ed8"
HEADER_LEFT   = "#1e3a8a"
HEADER_RIGHT  = "#4f8cf7"
FIELD_BORDER  = "#e3e8f7"
ENTRY_BG      = "#f4f6ff"
ENTRY_BORDER  = "#c9d3f5"
ICON_TILE_BG  = "#eef2ff"
TEXT_DARK     = "#1f2937"
GREEN         = "#1fa855"
GREEN_SOFT    = "#5fc785"
GREEN_SHADOW  = "#d3ecdd"
RED           = "#dc2626"
RED_SOFT      = "#e86b6b"
RED_SHADOW    = "#f5d5d5"

EMOJI_FONT = "Segoe UI Emoji"
UI_FONT = "Segoe UI"


# ============================================================
# 3. DRAWING HELPERS
# ============================================================
def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle on a canvas."""
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
           x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
           x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


def hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def blend(c1, c2, t):
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class RoundedButton(tk.Canvas):
    """A rounded (optionally dashed-outline) button drawn on a Canvas."""

    def __init__(self, parent, text, command, height=48, width=100, radius=14,
                 bg=BLUE, fg="white", hover=BLUE_HOVER, outline="", dash=None,
                 font=(UI_FONT, 13, "bold"), canvas_bg=CARD_BG):
        super().__init__(parent, width=width, height=height, bg=canvas_bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.text, self.command = text, command
        self.radius, self.fg = radius, fg
        self.base, self.hover = bg, hover
        self.outline, self.dash, self.font = outline, dash, font
        self.current = bg
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self._set(self.hover))
        self.bind("<Leave>", lambda e: self._set(self.base))
        self.bind("<Button-1>", lambda e: self.command())

    def _set(self, color):
        self.current = color
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        round_rect(self, 2, 2, w - 2, h - 2, self.radius, fill=self.current,
                   outline=self.outline, dash=self.dash, width=1.5)
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg, font=self.font)


class ResultBanner(tk.Canvas):
    """Big rounded result banner: green PLACED / red NOT PLACED / red error text."""

    def __init__(self, parent, height=96):
        super().__init__(parent, height=height, bg=CARD_BG, highlightthickness=0, bd=0)
        self.state, self.message = None, ""
        self.big = tkfont.Font(family=UI_FONT, size=30, weight="bold")
        self.bind("<Configure>", lambda e: self.render())

    def show(self, state, message=""):
        self.state, self.message = state, message
        self.render()

    def render(self):
        self.delete("all")
        if self.state is None:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if self.state == "placed":
            main, soft, shadow, icon, text = GREEN, GREEN_SOFT, GREEN_SHADOW, "\u2714", "PLACED"
        else:
            main, soft, shadow, icon, text = RED, RED_SOFT, RED_SHADOW, "\u2716", "NOT PLACED"

        # soft shadow + main banner
        round_rect(self, 6, 12, w - 6, h - 2, 22, fill=shadow, outline="")
        round_rect(self, 2, 2, w - 2, h - 10, 22, fill=main, outline="")
        cy = (2 + h - 10) / 2

        if self.state == "error":
            self.create_text(w / 2, cy, text=self.message, fill="white", width=w - 80,
                             font=(UI_FONT, 14, "bold"), justify="center")
            return

        d, gap = 58, 22
        tw = self.big.measure(text)
        sx = (w - (d + gap + tw)) / 2
        self.create_oval(sx, cy - d / 2, sx + d, cy + d / 2, fill=soft, outline="")
        self.create_text(sx + d / 2, cy, text=icon, fill="white", font=(UI_FONT, 22, "bold"))
        self.create_text(sx + d + gap, cy, text=text, fill="white", anchor="w", font=self.big)


# ============================================================
# 4. LOGIC (predict / reset / batch)
# ============================================================
def get_float(entry, field_name, errors):
    try:
        return float(entry.get())
    except ValueError:
        errors.append(field_name)
        return None


def predict_placement():
    errors = []
    cgpa = get_float(entry_cgpa, "CGPA", errors)
    backlogs = get_float(entry_backlogs, "Backlogs", errors)
    communication = get_float(entry_communication, "Communication", errors)
    projects = get_float(entry_projects, "Projects", errors)
    internships = get_float(entry_internships, "Internships", errors)
    technical = get_float(entry_technical, "Technical Skills", errors)

    if errors:
        banner.show("error", "Please check: " + ", ".join(errors))
        return

    # order must match FEATURE_COLUMNS used in training
    x_input = [[cgpa, backlogs, communication, projects, internships, technical]]
    prediction = classifier.predict(sc.transform(x_input))
    result = le.inverse_transform(prediction)[0]

    is_placed = str(result).strip().lower() == "placed"
    banner.show("placed" if is_placed else "not_placed")


def reset_fields():
    for e in (entry_cgpa, entry_backlogs, entry_communication,
              entry_projects, entry_internships, entry_technical):
        e.delete(0, tk.END)
    banner.show(None)


def batch_predict():
    input_path = filedialog.askopenfilename(title="Select student CSV file",
                                            filetypes=[("CSV files", "*.csv")])
    if not input_path:
        return
    try:
        batch_df = pd.read_csv(input_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the file:\n{e}")
        return

    missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
    if missing:
        messagebox.showerror("Missing columns",
                             "Your CSV must contain these columns:\n" + ", ".join(FEATURE_COLUMNS) +
                             "\n\nMissing: " + ", ".join(missing))
        return

    features = batch_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        messagebox.showerror("Invalid data",
                             "Some rows have empty or non-numeric values in the feature columns.\n"
                             "Please fix them and try again.")
        return

    predictions = le.inverse_transform(classifier.predict(sc.transform(features.values)))
    batch_df["Predicted_Placement"] = predictions

    save_path = filedialog.asksaveasfilename(title="Save predictions as", defaultextension=".csv",
                                             initialfile="placement_predictions.csv",
                                             filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return
    try:
        batch_df.to_csv(save_path, index=False)
    except Exception as e:
        messagebox.showerror("Error", f"Could not save the file:\n{e}")
        return

    total = len(predictions)
    placed = int((predictions == "Placed").sum())
    messagebox.showinfo("Batch prediction complete",
                        f"Students processed: {total}\nPredicted Placed: {placed}\n"
                        f"Predicted Not Placed: {total - placed}\n\nSaved to:\n{save_path}")


# ============================================================
# 5. GUI
# ============================================================
root = tk.Tk()
root.title("Student Placement Predictor")
root.geometry("1000x690")
root.minsize(900, 660)
root.configure(bg=BG)

card = tk.Frame(root, bg=CARD_BG, highlightbackground="#d3dbf5", highlightcolor="#d3dbf5", highlightthickness=1)
card.pack(fill="both", expand=True, padx=26, pady=26)

# ---------- Gradient header ----------
header = tk.Canvas(card, height=100, highlightthickness=0, bd=0, bg=NAVY)
header.pack(fill="x")


def draw_header(event=None):
    header.delete("all")
    w, h = header.winfo_width(), header.winfo_height()
    step = 3
    for px in range(0, w, step):
        color = blend(HEADER_LEFT, HEADER_RIGHT, px / max(w - 1, 1))
        header.create_rectangle(px, 0, px + step, h, fill=color, outline="")
    round_rect(header, 40, 18, 106, 84, 16, fill="#3f66c8", outline="#6f8fdc")
    header.create_text(73, 51, text="\U0001F393", font=(EMOJI_FONT, 26), fill="white")
    header.create_text(128, 38, text="Student Placement Predictor", anchor="w",
                       font=(UI_FONT, 24, "bold"), fill="white")
    header.create_text(130, 70, text="Predict your placement using Machine Learning", anchor="w",
                       font=(UI_FONT, 12), fill="#dce7ff")


header.bind("<Configure>", draw_header)

# ---------- Body ----------
body = tk.Frame(card, bg=CARD_BG)
body.pack(fill="both", expand=True, padx=32, pady=(22, 16))

title_row = tk.Frame(body, bg=CARD_BG)
title_row.pack(fill="x", pady=(0, 10))
tk.Frame(title_row, bg=BLUE, width=5, height=24).pack(side="left")
tk.Label(title_row, text="Enter Student Details", font=(UI_FONT, 15, "bold"),
         bg=CARD_BG, fg=NAVY).pack(side="left", padx=10)

grid = tk.Frame(body, bg=CARD_BG)
grid.pack(fill="x")
for c in range(3):
    grid.columnconfigure(c, weight=1, uniform="cols")


def add_field_card(row, col, icon, label_text):
    cell = tk.Frame(grid, bg=CARD_BG, highlightbackground=FIELD_BORDER, highlightcolor=FIELD_BORDER,
                    highlightthickness=1)
    cell.grid(row=row, column=col, sticky="ew", padx=7, pady=7)

    tile = tk.Frame(cell, bg=ICON_TILE_BG, width=46, height=46)
    tile.pack(side="left", padx=(12, 10), pady=14)
    tile.pack_propagate(False)
    tk.Label(tile, text=icon, font=(EMOJI_FONT, 18), bg=ICON_TILE_BG).pack(expand=True)

    right = tk.Frame(cell, bg=CARD_BG)
    right.pack(side="left", fill="x", expand=True, padx=(0, 12), pady=10)
    tk.Label(right, text=label_text, font=(UI_FONT, 9, "bold"), bg=CARD_BG,
             fg=TEXT_DARK, anchor="w").pack(fill="x")
    entry = tk.Entry(right, font=(UI_FONT, 14, "bold"), bg=ENTRY_BG, fg="#0f1b4d",
                     relief="flat", bd=0, insertbackground="#0f1b4d",
                     highlightthickness=2, highlightbackground=ENTRY_BORDER, highlightcolor=BLUE)
    entry.pack(fill="x", ipady=6, pady=(4, 0))
    return entry


entry_cgpa = add_field_card(0, 0, "\U0001F393", "CGPA")
entry_backlogs = add_field_card(0, 1, "\U0001F4DA", "Backlogs")
entry_communication = add_field_card(0, 2, "\U0001F4AC", "Communication Rating (1 - 10)")
entry_projects = add_field_card(1, 0, "\U0001F9E9", "Projects")
entry_internships = add_field_card(1, 1, "\U0001F4BC", "Internships")
entry_technical = add_field_card(1, 2, "\u2699\uFE0F", "Technical Skills (1 - 10)")

# ---------- Buttons ----------
btn_row = tk.Frame(body, bg=CARD_BG)
btn_row.pack(fill="x", pady=(16, 0), padx=7)
btn_row.columnconfigure(0, weight=1)

RoundedButton(btn_row, "Predict Placement", predict_placement, height=52, radius=16,
              bg=BLUE, hover=BLUE_HOVER).grid(row=0, column=0, sticky="ew", padx=(0, 14))
RoundedButton(btn_row, "Reset", reset_fields, height=52, width=200, radius=16,
              bg="white", fg=TEXT_DARK, hover="#f3f5fb", outline="#d5dbe8").grid(row=0, column=1)

RoundedButton(body, "Load CSV (Batch Predict)", batch_predict, height=46, radius=12,
              bg="#f1f5ff", fg="#1d4ed8", hover="#e3ebff", outline="#7c9cf0", dash=(5, 4),
              font=(UI_FONT, 11, "bold")).pack(fill="x", padx=7, pady=(12, 0))

# ---------- Result banner ----------
banner = ResultBanner(body, height=96)
banner.pack(fill="x", padx=7, pady=(16, 0))


root.mainloop()

