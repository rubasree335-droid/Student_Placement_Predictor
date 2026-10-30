import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox   # filedialog + messagebox added for batch prediction

dataset = pd.read_csv(r"E:\ds&ml\student_placement_dataset.csv")

FEATURE_COLUMNS = ["CGPA", "Backlogs", "Communication", "Projects", "Internships", "Technical_Skills"]
TARGET_COLUMN = "Placement_Status"

x = dataset[FEATURE_COLUMNS].values
y = np.asarray(dataset[TARGET_COLUMN], dtype=object)

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


BG            = "#eef1fb"
CARD_BG       = "#ffffff"
HEADER_BLUE   = "#1e3a8a"
ACCENT_BLUE   = "#2563eb"
TEXT_GRAY     = "#6b7280"
BORDER_GRAY   = "#e5e7eb"
GREEN_DARK    = "#16a34a"
RED_DARK      = "#dc2626"


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
        result_frame.config(bg=RED_DARK)
        result_label.config(text="Please check: " + ", ".join(errors), bg=RED_DARK, fg="white",
                             font=("Segoe UI", 11, "bold"))
        return

    x_input = [[cgpa, backlogs, communication, projects, internships, technical]]
    x_input_scaled = sc.transform(x_input)
    prediction = classifier.predict(x_input_scaled)
    result = le.inverse_transform(prediction)[0]

    is_placed = str(result).strip().lower() == "placed"
    banner_bg = GREEN_DARK if is_placed else RED_DARK
    display_text = "PLACED" if is_placed else "NOT PLACED"

    result_frame.config(bg=banner_bg)
    result_label.config(text=display_text, bg=banner_bg, fg="white", font=("Segoe UI", 22, "bold"))


def reset_fields():
    for e in (entry_cgpa, entry_backlogs, entry_communication, entry_projects,
              entry_internships, entry_technical):
        e.delete(0, tk.END)
    result_frame.config(bg=CARD_BG)
    result_label.config(text="", bg=CARD_BG)


# ---------------- NEW: batch prediction ----------------
def batch_predict():
    # 1. Ask the user to pick the CSV of students
    input_path = filedialog.askopenfilename(
        title="Select student CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not input_path:          # user closed the dialog without choosing
        return

    # 2. Read the file
    try:
        batch_df = pd.read_csv(input_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the file:\n{e}")
        return

    # 3. Check that all 6 required columns are present
    missing = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
    if missing:
        messagebox.showerror(
            "Missing columns",
            "Your CSV must contain these columns:\n" + ", ".join(FEATURE_COLUMNS) +
            "\n\nMissing: " + ", ".join(missing)
        )
        return

    # 4. Make sure every value is a valid number
    features = batch_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        messagebox.showerror(
            "Invalid data",
            "Some rows have empty or non-numeric values in the feature columns.\n"
            "Please fix them and try again."
        )
        return

    # 5. Scale with the SAME scaler used in training, then predict
    scaled = sc.transform(features.values)
    predictions = le.inverse_transform(classifier.predict(scaled))
    batch_df["Predicted_Placement"] = predictions

    # 6. Ask where to save the results
    save_path = filedialog.asksaveasfilename(
        title="Save predictions as",
        defaultextension=".csv",
        initialfile="placement_predictions.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not save_path:
        return

    try:
        batch_df.to_csv(save_path, index=False)
    except Exception as e:
        messagebox.showerror("Error", f"Could not save the file:\n{e}")
        return

    # 7. Show a summary
    total = len(predictions)
    placed = int((predictions == "Placed").sum())
    messagebox.showinfo(
        "Batch prediction complete",
        f"Students processed: {total}\n"
        f"Predicted Placed: {placed}\n"
        f"Predicted Not Placed: {total - placed}\n\n"
        f"Saved to:\n{save_path}"
    )
# -------------------------------------------------------


root = tk.Tk()
root.title("Student Placement Predictor")
root.geometry("420x660")          # slightly taller to fit the new button
root.configure(bg=BG)

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SUBTITLE = ("Segoe UI", 9)
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_ENTRY = ("Segoe UI", 10)


header_frame = tk.Frame(root, bg=BG)
header_frame.pack(fill="x", padx=20, pady=(20, 10))
tk.Label(header_frame, text="\U0001F393  Student Placement Predictor", font=FONT_TITLE,
         bg=BG, fg=HEADER_BLUE).pack(anchor="w")
tk.Label(header_frame, text="Predict your placement using Machine Learning",
         font=FONT_SUBTITLE, bg=BG, fg=TEXT_GRAY).pack(anchor="w")

card = tk.Frame(root, bg=CARD_BG, bd=1, relief="solid", highlightbackground=BORDER_GRAY)
card.pack(fill="both", expand=True, padx=20, pady=10)

tk.Label(card, text="Enter Student Details", font=("Segoe UI", 12, "bold"),
         bg=CARD_BG, fg=HEADER_BLUE, anchor="w").pack(fill="x", padx=20, pady=(18, 10))


def add_field(parent, label_text, placeholder=""):
    tk.Label(parent, text=label_text, font=FONT_LABEL, bg=CARD_BG, fg="#111827", anchor="w").pack(
        fill="x", padx=20, pady=(6, 2))
    entry = tk.Entry(parent, font=FONT_ENTRY, bg="white", relief="solid", bd=1, width=18,
                      highlightthickness=1, highlightbackground=BORDER_GRAY, highlightcolor=ACCENT_BLUE)
    entry.pack(anchor="w", padx=20, ipady=2)
    return entry


entry_cgpa = add_field(card, "CGPA")
entry_backlogs = add_field(card, "Backlogs")
entry_communication = add_field(card, "Communication Rating (1 - 10)")
entry_projects = add_field(card, "Projects")
entry_internships = add_field(card, "Internships")
entry_technical = add_field(card, "Technical Skills (1 - 10)")

button_frame = tk.Frame(card, bg=CARD_BG)
button_frame.pack(fill="x", padx=20, pady=(18, 8))

predict_btn = tk.Button(button_frame, text="Predict Placement", font=("Segoe UI", 11, "bold"),
                         bg=ACCENT_BLUE, fg="white", activebackground="#1d4ed8", activeforeground="white",
                         relief="flat", bd=0, command=predict_placement, cursor="hand2")
predict_btn.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))

reset_btn = tk.Button(button_frame, text="Reset", font=("Segoe UI", 11, "bold"),
                       bg="#f3f4f6", fg="#111827", activebackground="#e5e7eb",
                       relief="flat", bd=0, command=reset_fields, cursor="hand2")
reset_btn.pack(side="left", ipady=8, ipadx=15)

# ---------------- NEW: Load CSV button ----------------
batch_btn = tk.Button(card, text="Load CSV (Batch Predict)", font=("Segoe UI", 10, "bold"),
                       bg="#eff6ff", fg=ACCENT_BLUE, activebackground="#dbeafe",
                       relief="flat", bd=0, command=batch_predict, cursor="hand2")
batch_btn.pack(fill="x", padx=20, ipady=6)
# ------------------------------------------------------

result_frame = tk.Frame(card, bg=CARD_BG, height=70)
result_frame.pack(fill="x", padx=20, pady=(14, 20))
result_label = tk.Label(result_frame, text="", bg=CARD_BG)
result_label.pack(pady=18)

root.mainloop()