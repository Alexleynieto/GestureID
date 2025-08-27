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

# Config
PROJECT_ID = "gestureid"
LOCATION = "europe-west4"
TARGET_SIZE = 512
MODEL_NAME = "gemini-2.5-pro"
APP_BG = "#f0f2f5"
CARD_BG = "white"
ACCENT = "#4CAF50"

SYSTEM_PROMPT = (
    "Eres un sistema experto en reconocimiento gestual estático. Durante tu uso los usuarios te "
    "pasarán imágenes de manos realizando gestos del lenguaje de signos americano (ASL), del dataset "
    "de Kaggle. Entre estos encontramos 26 gestos de letras de la A a la Z, más 3 gestos adicionales, "
    "SPACE, DEL y NOTHING. IMPORTANTE: responde únicamente con las letras A-Z o los tokens SPACE, DEL, NOTHING."
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
    except Exception:
        return None


def call_vertex(png_bytes: bytes) -> Optional[str]:
    token = get_access_token()
    if not token:
        return None
    b64 = base64.b64encode(png_bytes).decode('utf-8')
    body = {
        "contents": [
            {"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/png", "data": b64}},
                {"text": "Identifica el gesto"}
            ]}
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.1, "candidateCount": 1, "maxOutputTokens": 16}
    }
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}:generateContent"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=45)
        if r.status_code != 200:
            return None
        data = r.json()
        cands = data.get("candidates")
        if isinstance(cands, list) and cands:
            c0 = cands[0]
            content = c0.get("content", {}) if isinstance(c0, dict) else {}
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list) and parts:
                txt = parts[0].get("text") if isinstance(parts[0], dict) else None
                return normalize_gesture(txt)
        return None
    except Exception:
        return None


def dummy_local_predict() -> str:
    return "NOTHING"


class SimpleGestureLoginApp:
    def __init__(self, root):
        self.root = root
        root.title("GestureID")
        try:
            root.attributes('-fullscreen', True)
            root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
        except Exception:
            try:
                root.state('zoomed')
            except Exception:
                root.geometry('1200x800')
        root.minsize(1000, 650)
        root.configure(bg=APP_BG)
        self._bg_photo = None
        self.cap = None
        self.running = False
        self.frame = None
        self.user_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.last_pred_var = tk.StringVar(value='-')
        self.mode = 'GESTOS'
        self._build_ui()
        self._bind_space()
        self.start_camera()

    def _build_ui(self):
        """Construye la interfaz principal con card centrada, panel derecha."""
        # Título
        tk.Label(self.root, text="GestureID", font=("Helvetica", 28, 'bold'), bg=self.root.cget('bg')).pack(pady=(12, 6))

        # Contenedor central con 3 columnas (0 espacio, 1 card, 2 cámara)
        self.container = tk.Frame(self.root, bg=self.root.cget('bg'))
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        # Config inicial (modo gestos por defecto)
        self._configure_columns_gestos()

        # CARD usuario y contraseña
        self.card = tk.Frame(self.container, bg=CARD_BG, bd=1, relief='groove', width=420, height=300)
        self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(10,0))
        self.card.grid_propagate(False)

        # Contenido card 
        self.label_user = ttk.Label(self.card, text="Usuario:", font=("Arial", 12))
        self.label_user.pack(anchor='w', pady=(12,0), padx=12)
        self.user_entry = ttk.Entry(self.card, textvariable=self.user_var, width=34, font=("Arial", 11))
        self.user_entry.pack(pady=(6,8), padx=12)
        self.label_pass = ttk.Label(self.card, text="Contraseña:", font=("Arial", 12))
        self.label_pass.pack(anchor='w', pady=(2,0), padx=12)
        # readonly en modo gestos, editable en modo PASS
        self.password_entry = ttk.Entry(self.card, textvariable=self.password_var, width=34, font=("Arial", 11), show='*', state='readonly')
        self.password_entry.pack(pady=(6,8), padx=12)

        # Botones de modo
        mode_frame = tk.Frame(self.card, bg=CARD_BG)
        mode_frame.pack(pady=(4,4))
        self.gesture_btn = tk.Button(mode_frame, text='Gestos', width=10, command=lambda: self.set_mode('GESTOS'))
        self.pass_btn = tk.Button(mode_frame, text='Contraseña', width=12, command=lambda: self.set_mode('PASS'))
        self.gesture_btn.pack(side=tk.LEFT, padx=6)
        self.pass_btn.pack(side=tk.LEFT, padx=6)

        # Acciones
        actions = tk.Frame(self.card, bg=CARD_BG)
        actions.pack(pady=(4,4))
        self.capture_btn = tk.Button(actions, text='Capturar (ESPACIO)', command=self.capture)
        self.capture_btn.pack(side=tk.LEFT, padx=(0,8))
        tk.Button(actions, text='Login', command=self.login).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='Salir', command=self.quit).pack(side=tk.LEFT, padx=4)

        # Último gesto
        pred_frame = tk.Frame(self.card, bg=CARD_BG)
        pred_frame.pack(pady=(6,8))
        tk.Label(pred_frame, text='Último gesto:', bg=CARD_BG, font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Label(pred_frame, textvariable=self.last_pred_var, bg=CARD_BG, font=("Arial", 14, 'bold'), fg='#333').pack(side=tk.LEFT, padx=6)

        # Panel de cámara
        self.panels_col = tk.Frame(self.container, bg=self.root.cget('bg'))
        self.webcam_holder = tk.Frame(self.panels_col, width=800, height=600, bg='#000')
        self.webcam_holder.pack_propagate(False)
        self.webcam_holder.pack(padx=(0,0), pady=(0,0))
        self.cam_label = tk.Label(self.webcam_holder, text='NO CAM', fg='white', bg='#000', font=('Helvetica', 16))
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        # Colocar panel
        self.panels_col.grid(row=0, column=2, padx=(12,20), pady=10, sticky='n')

        # Estilos y foco
        self._style_buttons(); self._refresh_mode(); self.user_entry.focus_set()
        for ms in (60, 200, 450):
            self.root.after(ms, self._refocus_user)
        self.root.after_idle(self._post_init_focus)

    def _configure_columns_gestos(self):
        """Columnas similares al fixed: col 0 (espacio), col 1 (card), col 2 (panel)."""
        for i in range(3):
            self.container.grid_columnconfigure(i, weight=0)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=0)
        self.container.grid_columnconfigure(2, weight=1)

    def _configure_columns_pass(self):
        # Tres columnas para centrar: card en la columna 1
        for i in range(3):
            self.container.grid_columnconfigure(i, weight=1, minsize=0)
        # Columna central puede tener peso 0 para evitar ensanche excesivo y centrar real
        self.container.grid_columnconfigure(1, weight=0)

    def _style_buttons(self):
        for btn in (getattr(self, 'gesture_btn', None), getattr(self, 'pass_btn', None), getattr(self, 'capture_btn', None)):
            if btn:
                try:
                    btn.configure(bg='#e9ecef', activebackground='#d0d4d8', bd=1, relief=tk.RAISED)
                except Exception:
                    pass

    def _refocus_user(self):
        """Asegura que el campo de usuario reciba el foco inicial.
        Se invoca varias veces con after para sobrevivir a dialogs de error de cámara."""
        if self.mode == 'GESTOS' and getattr(self, 'user_entry', None):
            try:
                self.user_entry.focus_set()
            except Exception:
                pass

    def _post_init_focus(self):
        """Forzar foco fuerte tras construcción completa y diálogos potenciales."""
        if getattr(self, 'user_entry', None):
            try:
                self.root.focus_force()
                self.user_entry.focus_set()
            except Exception:
                pass

    def _bind_space(self):
        # Usar bind en lugar de bind_all para no interferir con selección de Entry
        self.root.bind('<space>', self._on_space)

    def _on_space(self, event=None):
        if self.mode == 'GESTOS':
            self.capture()
            return 'break'  # Consumimos la barra espaciadora sólo en modo gestos
        # En modo contraseña dejamos que la pulsación se procese normalmente
        return None

    def set_mode(self, mode: str):
        if mode not in ('GESTOS', 'PASS'):
            return
        if self.mode == mode:
            return
        self.mode = mode
        self._refresh_mode()

    def _refresh_mode(self):
        active = {'bg': ACCENT, 'fg': 'white'}
        idle = {'bg': '#e0e0e0', 'fg': 'black'}
        if hasattr(self, 'gesture_btn'):
            if self.mode == 'GESTOS':
                self.gesture_btn.config(**active, relief=tk.SUNKEN)
                self.pass_btn.config(**idle, relief=tk.RAISED)
            else:
                self.pass_btn.config(**active, relief=tk.SUNKEN)
                self.gesture_btn.config(**idle, relief=tk.RAISED)
        if self.mode == 'GESTOS':
            # Mostrar card centrada (col 1) y panel (col 2) como en fixed
            self._configure_columns_gestos()
            if hasattr(self, 'panels_col'):
                try:
                    self.panels_col.grid_forget()
                    self.panels_col.grid(row=0, column=2, padx=(12,20), pady=10, sticky='n')
                except Exception:
                    pass
            if hasattr(self, 'card'):
                try:
                    self.card.grid_forget()
                    self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(10,0))
                except Exception:
                    pass
            self.capture_btn.config(state='normal')
            self.password_entry.configure(state='readonly')
            if hasattr(self, 'user_entry'):
                try:
                    self.user_entry.focus_set()
                except Exception:
                    pass
        else:
            # Ocultar panel y centrar card (col 1) con columnas simétricas
            self._configure_columns_pass()
            if hasattr(self, 'panels_col'):
                try:
                    self.panels_col.grid_forget()
                except Exception:
                    pass
            if hasattr(self, 'card'):
                try:
                    self.card.grid_forget(); self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(40,10))
                except Exception:
                    pass
            self.capture_btn.config(state='disabled')
            self.password_entry.configure(state='normal')
            try:
                self.password_entry.focus_set()
            except Exception:
                pass

    def start_camera(self):
        if self.cap is None:
            attempts = [(0, cv2.CAP_MSMF),(0, cv2.CAP_DSHOW),(0, cv2.CAP_VFW),(0, cv2.CAP_ANY),(1, cv2.CAP_ANY),(0,0)]
            for idx, backend in attempts:
                try:
                    cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                    time.sleep(0.25)
                    if cap.isOpened():
                        self.cap = cap; print(f"[Camera] abierta (idx={idx}, backend={backend})"); break
                    cap.release()
                except Exception:
                    pass
            if self.cap is None:
                # Mostrar popup de error después de que la UI haya aparecido para no robar foco inicial
                self.root.after(300, lambda: (messagebox.showerror('Cámara', 'No se pudo abrir la cámara'), self._refocus_user()))
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

            imgtk = ImageTk.PhotoImage(pil.resize((800, 600)))
            def upd():
                self.cam_label.config(image=imgtk, text='')
                self.cam_label.imgtk = imgtk
            try:
                self.root.after(0, upd)
            except Exception:
                pass
            time.sleep(1/30)

    def capture(self):
        if self.mode != 'GESTOS' or self.frame is None:
            return
        pil_img = letterbox_512(self.frame)
        png_bytes = pil_to_png_bytes(pil_img)
        self.last_pred_var.set('...')
        threading.Thread(target=self._infer_thread, args=(png_bytes,), daemon=True).start()

    def _infer_thread(self, png_bytes: bytes):
        pred = call_vertex(png_bytes)
        if pred is None:
            pred = dummy_local_predict()
        if pred and pred != 'NOTHING':
            pwd = self.password_var.get()
            if pred == 'DEL': pwd = pwd[:-1]
            elif pred == 'SPACE': pwd += ' '
            else: pwd += pred
            self.password_var.set(pwd)
        self.root.after(0, lambda: self.last_pred_var.set(pred))

    def login(self):
        user = self.user_var.get().strip(); pwd = self.password_var.get()
        if not user:
            messagebox.showerror('Login', 'Introduce un usuario'); return
        if not pwd:
            messagebox.showwarning('Login', 'No hay contraseña (gestos o escrita)'); return
        origen = 'gestos' if self.mode == 'GESTOS' else 'alfanumérico'
        mask = '*' * len(pwd)
        messagebox.showinfo('Login', f'Usuario: {user}\nPassword: {mask}\n(Método: {origen} – validación simulada)')

    def quit(self):
        self.running = False
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        try:
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGestureLoginApp(root)

    root.mainloop()
