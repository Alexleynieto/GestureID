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

# ----------------- Config -----------------
PROJECT_ID = "gestureid"
LOCATION = "europe-west4"
TARGET_SIZE = 512
MODEL_NAME = "gemini-2.5-pro"

PROMPT = (
    "Eres un sistema experto en reconocimiento gestual estático. Durante tu uso los usuarios te "
    "pasarán imágenes de manos realizando gestos del lenguaje de signos americano (ASL), del dataset "
    "de Kaggle. Entre estos encontramos 26 gestos de letras de la A a la Z, más 3 gestos adicionales, "
    "SPACE, DEL y NOTHING. IMPORTANTE: responde únicamente con las letras A-Z o los tokens SPACE, DEL, NOTHING."
)

ACCENT = "#4CAF50"


def letterbox_to_512_from_bgr(frame: np.ndarray, target: int = TARGET_SIZE) -> Image.Image:
    """Redimensiona con letterbox a 512x512 como en Gesture_ID.py."""
    h, w = frame.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def get_access_token_adc() -> Optional[str]:
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception as e:
        print("ADC token error:", e)
        return None

def extract_text_from_known_paths(parsed):
    try:
        if not isinstance(parsed, dict):
            return None
        candidates = parsed.get("candidates")
        if isinstance(candidates, list) and len(candidates) > 0:
            first_c = candidates[0]
            if isinstance(first_c, dict):
                content = first_c.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list) and len(parts) > 0:
                        first_part = parts[0]
                        if isinstance(first_part, dict) and "text" in first_part:
                            t = first_part.get("text")
                            if isinstance(t, str):
                                return t
    except Exception:
        pass
    return None
    
def find_first_string_in_json(el):
    if el is None:
        return None
    if isinstance(el, dict):
        t = el.get("text")
        if isinstance(t, str) and t.strip():
            return t
        if "parts" in el and isinstance(el["parts"], list):
            for p in el["parts"]:
                found = find_first_string_in_json(p)
                if found:
                    return found
        for k, v in el.items():
            if k.lower() == "role":
                continue
            found = find_first_string_in_json(v)
            if found:
                return found
    elif isinstance(el, list):
        for item in el:
            found = find_first_string_in_json(item)
            if found:
                return found
    elif isinstance(el, str):
        if el.strip():
            return el
    return None

def normalize_gesture(raw: Optional[str]) -> str:
    """Extrae un token permitido de una respuesta libre del modelo.

    Reglas:
    1. Prioriza tokens exactos SPACE / DEL / NOTHING.
    2. Detecta patrones 'letter is X', 'letra: X', 'sign: X'.
    3. Una única letra aislada en el texto -> devuelve.
    4. Si todas las letras (hasta 4) son iguales -> devuelve esa letra.
    5. Fallback: última letra tras 'IS'.
    """
    if raw is None:
        return "NOTHING"
    s_full = raw.strip()
    if not s_full:
        return "NOTHING"
    s = s_full.upper()
    if "SPACE" in s:
        return "SPACE"
    if "DEL" in s:
        return "DEL"
    if s == 'NOTHING':
        return 'NOTHING'
    import re
    pat = re.search(r"(?:LETTER|LETRA|SIGN|SIGNO)[^A-Z]*([A-Z])\b", s)
    if pat:
        c = pat.group(1)
        if 'A' <= c <= 'Z':
            return c
    isolated = re.findall(r"\b([A-Z])\b", s)
    if len(isolated) == 1 and 'A' <= isolated[0] <= 'Z':
        return isolated[0]
    letters = re.findall(r"[A-Z]", s)
    uniq = set(letters)
    if len(letters) == 1:
        return letters[0]
    if len(uniq) == 1 and 1 < len(letters) <= 4:
        return letters[0]
    trail = re.search(r"\bIS\s+([A-Z])\b", s)
    if trail:
        c = trail.group(1)
        if 'A' <= c <= 'Z':
            return c
    return "NOTHING"

def call_vertex_generate(png_bytes: bytes) -> Optional[str]:
    """Llama al modelo Gemini (Vertex) y devuelve token normalizado."""
    token = get_access_token_adc()
    if not token:
        print("No ADC token")
        return None
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    user_image_part = {"inlineData": {"mimeType": "image/png", "data": b64}}
    user_text_part = {"text": "Identifica el gesto de la imagen."}
    parts = [user_image_part, user_text_part]
    content_wrapper = {"role": "user", "parts": parts}
    contents = [content_wrapper]
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "candidateCount": 1,
            "maxOutputTokens": 4096,
        },
        "systemInstruction": {"parts": [{"text": PROMPT}]},
    }
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}:generateContent"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
    except Exception as e:
        print("Request error:", e)
        return None
    if resp.status_code != 200:
        print("Vertex API error", resp.status_code, resp.text[:200])
        return None
    try:
        parsed = resp.json()
    except Exception:
        return normalize_gesture(resp.text)
    extracted = extract_text_from_known_paths(parsed)
    if extracted:
        print(f"[Vertex] primary text: {extracted!r}")
        return normalize_gesture(extracted)
    found = find_first_string_in_json(parsed)
    if found:
        print(f"[Vertex] fallback text: {found!r}")
        return normalize_gesture(found)
    dumped = json.dumps(parsed)
    print(f"[Vertex] dump length={len(dumped)}")
    return normalize_gesture(dumped)


def call_local_model(pil_img: Image.Image) -> Optional[str]:
    return "NOTHING"


class GestureLoginApp:
    """Versión simplificada: dos modos (Gestos/Contraseña), cámara y predicción."""
    def __init__(self, root: tk.Tk):
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
        root.minsize(900, 600)
        root.configure(bg="#f0f2f5")

        # Estado
        self.cap = None
        self.frame = None
        self.running = False
        self.mode = 'GESTOS'

        # Vars
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.last_pred_var = tk.StringVar(value='-')

        # Tamaños
        self.cam_size = (800, 600)

        self.build_ui()
        self._bind_space()
        self.start_camera()
        try:
            self.username_entry.focus_set()
        except Exception:
            pass
    
    # ---------------- UI -----------------

    def build_ui(self):
        tk.Label(self.root, text="GestureID", font=("Helvetica", 28, 'bold'), bg=self.root.cget('bg')).pack(pady=(12,6))

        self.container = tk.Frame(self.root, bg=self.root.cget('bg'))
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        for i in range(3):
            self.container.grid_columnconfigure(i, weight=1)

        # Card
        self.card = tk.Frame(self.container, bg='white', bd=1, relief='groove', width=420, height=300)
        self.card.grid_propagate(False)
        self.card.grid(row=0, column=1, sticky='n', padx=6, pady=(10,0))

        ttk.Label(self.card, text='Usuario:', font=('Arial',12)).pack(anchor='w', padx=12, pady=(10,0))
        self.username_entry = ttk.Entry(self.card, textvariable=self.username_var, width=34)
        self.username_entry.pack(padx=12, pady=(6,8))
        self.pass_label = ttk.Label(self.card, text='Contraseña gestual:', font=('Arial',12))
        self.pass_label.pack(anchor='w', padx=12, pady=(2,0))
        self.password_entry = ttk.Entry(self.card, textvariable=self.password_var, width=34, show='*', state='readonly')
        self.password_entry.pack(padx=12, pady=(6,8))

        mode_row = tk.Frame(self.card, bg='white')
        mode_row.pack(pady=(4,4))
        self.gesture_btn = tk.Button(mode_row, text='Gestos', width=12, command=lambda: self.set_mode('GESTOS'))
        self.pass_btn = tk.Button(mode_row, text='Contraseña', width=12, command=lambda: self.set_mode('CONTRASEÑA'))
        self.gesture_btn.pack(side=tk.LEFT, padx=6)
        self.pass_btn.pack(side=tk.LEFT, padx=6)

        actions = tk.Frame(self.card, bg='white')
        actions.pack(pady=(4,4))
        self.capture_btn = tk.Button(actions, text='Capturar (Espacio)', command=self.capture)
        self.capture_btn.pack(side=tk.LEFT, padx=(0,8))
        tk.Button(actions, text='Login', command=self.login).pack(side=tk.LEFT, padx=4)
        tk.Button(actions, text='Salir', command=self.quit).pack(side=tk.LEFT, padx=4)

        pred_frame = tk.Frame(self.card, bg='white')
        pred_frame.pack(pady=(6,4))
        tk.Label(pred_frame, text='Último gesto:', bg='white').pack(side=tk.LEFT)
        tk.Label(pred_frame, textvariable=self.last_pred_var, bg='white', font=('Arial',14,'bold')).pack(side=tk.LEFT, padx=6)

        # Cámara
        self.webcam_holder = tk.Frame(self.container, width=self.cam_size[0], height=self.cam_size[1], bg='#000')
        self.webcam_holder.grid(row=0, column=2, sticky='n', padx=(12,20), pady=10)
        self.webcam_holder.pack_propagate(False)
        self.webcam_label = tk.Label(self.webcam_holder, text='NO CAM', fg='white', bg='#000', font=('Helvetica',14))
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        self._style_mode_buttons()
        self._refresh_mode_buttons()


    def _style_mode_buttons(self):
        for btn in (self.gesture_btn, self.pass_btn):
            try:
                btn.configure(bg='#e6e6e6', activebackground='#d0d0d0', bd=1, relief=tk.RAISED)
            except Exception:
                pass

    def _refresh_mode_buttons(self):
        active = {'bg': ACCENT, 'fg': 'white', 'relief': tk.SUNKEN}
        idle = {'bg': '#e6e6e6', 'fg': 'black', 'relief': tk.RAISED}
        if self.mode == 'GESTOS':
            self.gesture_btn.config(**active)
            self.pass_btn.config(**idle)
            self.password_entry.configure(state='readonly')
            self.pass_label.config(text='Contraseña gestual:')
        else:
            self.pass_btn.config(**active)
            self.gesture_btn.config(**idle)
            self.password_entry.configure(state='normal')
            self.pass_label.config(text='Contraseña:')
    
    def set_mode(self, mode: str):
        if mode not in ('GESTOS','CONTRASEÑA'):
            return
        if self.mode == mode:
            return
        self.mode = mode
        # limpiar contraseña al cambiar
        try: self.password_var.set('')
        except Exception: pass
        if mode == 'CONTRASEÑA':
            # permitir escritura manual y parar cámara
            self.password_entry.configure(state='normal')
            try:
                self.webcam_holder.grid_remove()
            except Exception:
                pass
            self.stop_camera()
            try:
                self.capture_btn.config(state='disabled')
            except Exception:
                pass
        else:
            # modo gestos: activar cámara y ocultar edición directa
            self.password_entry.configure(state='readonly')
            try:
                self.webcam_holder.grid()
            except Exception:
                pass
            if not self.running:
                self.start_camera()
            try:
                self.capture_btn.config(state='normal')
            except Exception:
                pass
        self._refresh_mode_buttons()


    # ------ Captura de Gestos ------

    def _on_space(self, event=None):
        if self.mode == 'GESTOS':
            # Consumimos la barra espaciadora sólo en modo gestos
            self.capture()
            return 'break'  
        
        # En modo contraseña dejamos que la pulsación se procese normalmente
        return None

    def _bind_space(self):
        try:
            self.root.bind('<space>', self._on_space)
        except Exception:
            pass

    # (Se eliminaron definiciones duplicadas de set_mode/_refresh_mode)

    # ---------------- Cámara & Captura -----------------
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
            imgtk = ImageTk.PhotoImage(pil.resize(self.cam_size))
            def upd():
                try:
                    self.webcam_label.config(image=imgtk, text='')
                    self.webcam_label.imgtk = imgtk
                except Exception:
                    pass
            self.root.after(0, upd)
            time.sleep(1/30)

    def stop_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def capture(self):
        if self.mode != 'GESTOS' or self.frame is None:
            return
        letterboxed = letterbox_to_512_from_bgr(self.frame)
        png_bytes = pil_to_png_bytes(letterboxed)
        self.last_pred_var.set('...')
        threading.Thread(target=self._infer_thread, args=(png_bytes,), daemon=True).start()

    def _infer_thread(self, png_bytes: bytes):
        pred = call_vertex_generate(png_bytes)
        if pred is None:
            pred = 'NOTHING'
        if pred and pred != 'NOTHING':
            cur = self.password_var.get()
            if pred == 'DEL':
                cur = cur[:-1]
            elif pred == 'SPACE':
                cur += ' '
            else:
                cur += pred
            self.password_var.set(cur)
        self.root.after(0, lambda: self.last_pred_var.set(pred))

    def login(self):
        user = self.username_var.get().strip()
        pwd = self.password_var.get()
        if not user:
            messagebox.showerror('Login', 'Introduce un usuario'); return
        if not pwd:
            messagebox.showwarning('Login', 'No hay contraseña'); return
        origen = 'gestos' if self.mode == 'GESTOS' else 'texto'
        messagebox.showinfo('Login', f'Usuario: {user}\nPassword: {"*"*len(pwd)}\n(Método: {origen} – simulación)')

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
    app = GestureLoginApp(root)
    root.mainloop()
