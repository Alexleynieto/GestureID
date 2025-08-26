import base64
import io
import json
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import requests
import google.auth
from google.auth.transport.requests import Request

# Config mínima
PROJECT_ID = "gestureid"
LOCATION = "europe-west4"
TARGET_SIZE = 512
MODEL_NAME = "gemini-2.5-pro"  # Se puede dejar fijo

SYSTEM_PROMPT = (
    "Eres un sistema que reconoce un único gesto estático de ASL. Responde sólo con A-Z, SPACE, DEL o NOTHING."
)


def letterbox_512(frame: np.ndarray) -> Image.Image:
    h, w = frame.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    x = (TARGET_SIZE - nw) // 2
    y = (TARGET_SIZE - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize_gesture(raw: Optional[str]) -> str:
    if not raw:
        return "NOTHING"
    s = raw.strip().upper()
    if "SPACE" in s:
        return "SPACE"
    if "DEL" in s:
        return "DEL"
    letters = ''.join([c for c in s if 'A' <= c <= 'Z'])
    if len(letters) == 1:
        return letters
    return "NOTHING"


def get_access_token() -> Optional[str]:
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception as e:
        print("[Vertex] Token error:", e)
        return None


def call_vertex(png_bytes: bytes) -> Optional[str]:
    token = get_access_token()
    if not token:
        return None
    b64 = base64.b64encode(png_bytes).decode('utf-8')
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": b64}},
                    {"text": "Identifica el gesto"}
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.1, "candidateCount": 1, "maxOutputTokens": 16}
    }
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}:generateContent"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=45)
        if r.status_code != 200:
            print("[Vertex] HTTP", r.status_code, r.text[:200])
            return None
        data = r.json()
        # Ruta típica: candidates[0].content.parts[0].text
        cands = data.get("candidates")
        if isinstance(cands, list) and cands:
            c0 = cands[0]
            content = c0.get("content", {}) if isinstance(c0, dict) else {}
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list) and parts:
                txt = parts[0].get("text") if isinstance(parts[0], dict) else None
                return normalize_gesture(txt)
        # fallback dump
        return normalize_gesture(json.dumps(data)[:30])
    except Exception as e:
        print("[Vertex] error:", e)
        return None


def dummy_local_predict() -> str:
    # Siempre NOTHING; reemplazar con modelo local en el futuro.
    return "NOTHING"


class SimpleGestureLoginApp:
    def __init__(self, root):
        self.root = root
        root.title("GestureID - Simple")
        root.geometry("900x600")
        root.configure(bg="#f5f5f5")

        self.cap = None
        self.running = False
        self.frame = None
        self.sequence_var = tk.StringVar(value="")
        self.user_var = tk.StringVar()
        self.last_pred_var = tk.StringVar(value="-")

        self._build_ui()
        self._bind_space()
        self.start_camera()

    def _build_ui(self):
        top = tk.Frame(self.root, bg=self.root.cget("bg"))
        top.pack(pady=12)
        tk.Label(top, text="Login Gestual (Demo Inicial)", font=("Arial", 20, "bold"), bg=self.root.cget("bg")).pack()

        form = tk.Frame(self.root, bg=self.root.cget("bg"))
        form.pack(pady=8)
        tk.Label(form, text="Usuario:", font=("Arial", 12), bg=self.root.cget("bg")).grid(row=0, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(form, textvariable=self.user_var, width=28).grid(row=0, column=1, padx=4, pady=4)

        tk.Label(form, text="Secuencia gestos:", font=("Arial", 12), bg=self.root.cget("bg")).grid(row=1, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(form, textvariable=self.sequence_var, width=28, state="readonly").grid(row=1, column=1, padx=4, pady=4)

        btns = tk.Frame(self.root, bg=self.root.cget("bg"))
        btns.pack(pady=6)
        tk.Button(btns, text="Capturar (ESPACIO)", width=18, command=self.capture).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Borrar", width=10, command=lambda: self.sequence_var.set(""))
        tk.Button(btns, text="Borrar", width=10, command=lambda: self.sequence_var.set(""))
        # Remove duplicate button by only packing once
        btns.pack_forget()
        btns.pack(pady=6)
        # Re-add buttons cleanly
        for w in list(btns.winfo_children()):
            w.destroy()
        tk.Button(btns, text="Capturar (ESPACIO)", width=18, command=self.capture).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Borrar", width=10, command=lambda: self.sequence_var.set(""))
        tk.Button(btns, text="Login", width=10, command=self.login).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Salir", width=10, command=self.quit).pack(side=tk.LEFT, padx=6)

        pred_frame = tk.Frame(self.root, bg=self.root.cget("bg"))
        pred_frame.pack(pady=4)
        tk.Label(pred_frame, text="Último gesto:", bg=self.root.cget("bg"), font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Label(pred_frame, textvariable=self.last_pred_var, bg=self.root.cget("bg"), font=("Arial", 14, "bold"), fg="#333").pack(side=tk.LEFT, padx=6)

        # Contenedor cámara
        self.cam_holder = tk.Frame(self.root, width=640, height=480, bg="#000")
        self.cam_holder.pack(pady=12)
        self.cam_holder.pack_propagate(False)
        self.cam_label = tk.Label(self.cam_holder, text="CAMERA", fg="white", bg="#000")
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        note = tk.Label(self.root, text="(Esta demo no valida usuario real. Sólo acumula gestos.)", bg=self.root.cget("bg"), fg="#555")
        note.pack(pady=6)

    def _bind_space(self):
        self.root.bind_all('<space>', self._on_space)

    def _on_space(self, event=None):
        self.capture()
        return 'break'

    # Cámara
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Cámara", "No se pudo abrir la cámara")
                self.cap.release()
                self.cap = None
                return
        if not self.running:
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(pil.resize((640, 480)))
            def update():
                self.cam_label.config(image=imgtk, text='')
                self.cam_label.imgtk = imgtk
            try:
                self.root.after(0, update)
            except Exception:
                pass
            time.sleep(1/30)

    # Captura
    def capture(self):
        if self.frame is None:
            return
        # Preparar imagen letterbox y bytes
        pil_img = letterbox_512(self.frame)
        png_bytes = pil_to_png_bytes(pil_img)
        self.last_pred_var.set("...")
        threading.Thread(target=self._infer_thread, args=(png_bytes,), daemon=True).start()

    def _infer_thread(self, png_bytes: bytes):
        # Intentar Vertex, si falla usamos stub local
        pred = call_vertex(png_bytes)
        if pred is None:
            pred = dummy_local_predict()
        if pred == 'NOTHING':
            pass  # no agrega
        else:
            current = self.sequence_var.get()
            self.sequence_var.set(current + pred)
        def fin():
            self.last_pred_var.set(pred)
        self.root.after(0, fin)

    def login(self):
        user = self.user_var.get().strip()
        seq = self.sequence_var.get()
        if not user:
            messagebox.showerror("Login", "Introduce un usuario")
            return
        if not seq:
            messagebox.showwarning("Login", "Captura al menos un gesto")
            return
        messagebox.showinfo("Login", f"Usuario: {user}\nGestos: {seq}\n(Validación simulada)")

    def quit(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGestureLoginApp(root)
    root.mainloop()