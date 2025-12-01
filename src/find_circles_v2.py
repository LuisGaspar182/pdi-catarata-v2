import os
import cv2
import numpy as np
import math


# ============================================================
# CONFIG
# ============================================================

SCALE = 0.40     # reduzir imagem para acelerar
SECTOR_ANGLE_START = 20
SECTOR_ANGLE_END   = 160


# ============================================================
# REDIMENSÃO
# ============================================================

def redimensionar(img):
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

def reverter_ponto_small_to_full(pt):
    return int(pt[0] / SCALE), int(pt[1] / SCALE)

def reverter_contorno_small_to_full(cnt):
    return np.array([reverter_ponto_small_to_full(p[0]) for p in cnt]).reshape(-1,1,2)



# ============================================================
# KALMAN FILTER
# ============================================================

def criar_kalman():
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1.,0.,1.,0.],
        [0.,1.,0.,1.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf

kalman = criar_kalman()
kalman_initialized = False



# ============================================================
# INICIALIZAÇÃO (HSV)
# ============================================================

def inicializar_centro_hsv(img_small):
    """
    Inicializa o centro da íris usando segmentação por cor azul.
    Extremamente robusto para este dataset.
    """
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    azul_low  = np.array([85, 20, 20])
    azul_high = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, azul_low, azul_high)

    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return None

    iris = max(conts, key=cv2.contourArea)

    M = cv2.moments(iris)
    if M["m00"] == 0:
        return None

    cx = M["m10"]/M["m00"]
    cy = M["m01"]/M["m00"]

    area = cv2.contourArea(iris)
    r_est = int(np.sqrt(area/np.pi))

    return int(cx), int(cy), r_est



# ============================================================
# DETECÇÃO DA ÍRIS (HSV)
# ============================================================

def detectar_iris_hsv(img_small):
    """
    Detecta a íris REAL com HSV, evitando protuberâncias ligadas por 'pontes' finas.
    - Faz abertura/fechamento com kernel grande para quebrar conexões finas
    - Mantém apenas o maior componente
    - Suaviza/regulariza o contorno via convexHull e tentativa de fitEllipse
    """
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    azul_low  = np.array([85, 20, 20])
    azul_high = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, azul_low, azul_high)

    # 1) Limpeza forte para quebrar pontes/ligacoes finas (kernel maior)
    kernel_close = np.ones((11,11), np.uint8)
    kernel_open  = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # fecha buracos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)    # remove pontes finas

    # 2) Isolar apenas o maior componente azul (íris) — evita pequenas manchas azuis
    conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return None, mask

    # escolher maior componente (área)
    conts_sorted = sorted(conts, key=cv2.contourArea, reverse=True)
    iris_raw = conts_sorted[0]

    # 3) criar uma máscara somente do maior componente e reprocessar para garantir integridade
    mask_main = np.zeros_like(mask)
    cv2.drawContours(mask_main, [iris_raw], -1, 255, -1)

    # apagar pequenas protuberancias com uma erosao leve e depois dilatar
    kernel_small = np.ones((7,7), np.uint8)
    mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_ERODE, kernel_small, iterations=1)
    mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_DILATE, kernel_small, iterations=1)

    # 4) extrair contorno corrigido
    conts2, _ = cv2.findContours(mask_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts2:
        return None, mask_main

    iris_cont = max(conts2, key=cv2.contourArea)

    # 5) suavizar / regularizar: preferir fitEllipse se possível, senão convex hull
    try:
        if len(iris_cont) >= 5:
            ellipse = cv2.fitEllipse(iris_cont)
            # converter ellipse em contorno aproximado para desenhar/usar adiante
            (cx,cy),(MA,ma),angle = ellipse
            # gerar contorno elíptico como pontos
            pts = []
            for t in np.linspace(0, 2*np.pi, 180):
                x = int(cx + (MA/2)*np.cos(t)*np.cos(np.deg2rad(angle)) - (ma/2)*np.sin(t)*np.sin(np.deg2rad(angle)))
                y = int(cy + (MA/2)*np.cos(t)*np.sin(np.deg2rad(angle)) + (ma/2)*np.sin(t)*np.cos(np.deg2rad(angle)))
                pts.append([ [x,y] ])
            iris_smooth = np.array(pts, dtype=np.int32)
            iris_cont = iris_smooth
        else:
            # fallback: convex hull (remove pequenos entalhes)
            hull = cv2.convexHull(iris_cont)
            iris_cont = hull
    except Exception:
        hull = cv2.convexHull(iris_cont)
        iris_cont = hull

    return iris_cont, mask_main


# ============================================================
# DETECÇÃO DA PINÇA (Sobel + setor)
# ============================================================

def detectar_pinca_sobel_setor(img_small, centro_small, r_est):
    """
    Detecta a pinça combinando Sobel (bordas) com uma máscara de 'metal' (baixa saturação, alto valor).
    Permite detectar instrumentação metálica mesmo quando parcialmente dentro do disco azul.
    """
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # 1) Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.convertScaleAbs(mag)

    # 2) Máscara de metal: baixa saturação, alto valor (ajuste conforme seu vídeo)
    # metal tende a ser pouco saturado e brilhante (reflexo)
    metal_low  = np.array([0, 0, 120])
    metal_high = np.array([180, 80, 255])
    mask_metal = cv2.inRange(hsv, metal_low, metal_high)

    # 3) bordas fortes + metal
    _, mask_edges = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)  # threshold mais suave
    mask_comb = cv2.bitwise_and(mask_edges, mask_metal)

    # 4) limitar ao sector onde a pinça costuma aparecer (evita ruído)
    cx, cy = int(centro_small[0]), int(centro_small[1])
    mask_sector = np.zeros_like(mask_comb)
    ang0 = np.deg2rad(SECTOR_ANGLE_START)
    ang1 = np.deg2rad(SECTOR_ANGLE_END)
    pts = [(cx, cy)]
    # usar r_est*1.2 para cobrir área maior
    for ang in np.linspace(ang0, ang1, num=60):
        px = int(cx + r_est * 1.2 * math.cos(ang))
        py = int(cy + r_est * 1.2 * math.sin(ang))
        pts.append((px, py))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask_sector, [pts], 255)
    mask_comb = cv2.bitwise_and(mask_comb, mask_sector)

    # 5) morfologia para juntar as bordas e eliminar ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask_comb = cv2.morphologyEx(mask_comb, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_comb = cv2.morphologyEx(mask_comb, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) contornos e escolha por forma alongada
    conts, _ = cv2.findContours(mask_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    for c in conts:
        area = cv2.contourArea(c)
        if area < 40:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ratio = max(w/h, h/w)
        score = ratio * area
        # instrument tends to be elongated and moderate/large area
        if ratio > 2.5 and area > 150 and score > best_score:
            best_score = score
            best = c

    # 7) fallback: se não encontrar via metal+sobel, tentar só via sobel com threshold menor
    if best is None:
        _, mask_edges2 = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
        mask_edges2 = cv2.bitwise_and(mask_edges2, mask_sector)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask_edges2 = cv2.morphologyEx(mask_edges2, cv2.MORPH_CLOSE, kernel2, iterations=1)
        conts2, _ = cv2.findContours(mask_edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts2:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            if w==0 or h==0:
                continue
            ratio = max(w/h, h/w)
            if ratio > 2.0 and area > 120:
                best = c
                break

    return best, mask_comb


# ============================================================
# DESENHAR RESULTADOS
# ============================================================

def pintar_saida(img, iris_cont_small, pinca_cont_small, centro_small):

    out = img.copy()

    if iris_cont_small is not None:
        iris_full = reverter_contorno_small_to_full(iris_cont_small)
        cv2.drawContours(out, [iris_full], -1, (0,255,0), 3)

        overlay = out.copy()
        cv2.drawContours(overlay, [iris_full], -1, (0,255,0), -1)
        out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

    if pinca_cont_small is not None:
        pinca_full = reverter_contorno_small_to_full(pinca_cont_small)
        overlay = out.copy()
        cv2.drawContours(overlay, [pinca_full], -1, (0,255,255), -1)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    cx, cy = reverter_ponto_small_to_full(centro_small)
    cv2.circle(out, (cx, cy), 7, (0,0,255), -1)
    cv2.circle(out, (cx, cy), 16, (0,0,0), 2)

    cv2.putText(out, "Iris (HSV)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(out, "Pinca (Sobel setor)", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(out, "Centro (Kalman)", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    return out



# ============================================================
# PROCESSAR UM FRAME
# ============================================================

def processar_frame(filepath):
    global kalman_initialized, kalman

    img = cv2.imread(filepath)
    if img is None:
        return None

    img_small = redimensionar(img)

    # ---------- Inicialização HSV (1 vez) ----------
    if not kalman_initialized:
        init = inicializar_centro_hsv(img_small)
        if init is not None:
            cx0, cy0, r0 = init
        else:
            h, w = img_small.shape[:2]
            cx0, cy0 = w//2, h//2

        kalman.statePost = np.array([[float(cx0)],
                                     [float(cy0)],
                                     [0.],
                                     [0.]], dtype=np.float32)

        kalman_initialized = True

    # ---------- Previsão ----------
    pred = kalman.predict()
    cx_pred, cy_pred = float(pred[0][0]), float(pred[1][0])

    # ---------- ÍRIS (via HSV) ----------
    iris_cont_small, mask_iris = detectar_iris_hsv(img_small)

    # ---------- Centro da íris via momentos ----------
    centro_detectado_small = None

    if iris_cont_small is not None:
        M = cv2.moments(iris_cont_small)
        if M["m00"] != 0:
            centro_detectado_small = (M["m10"]/M["m00"], M["m01"]/M["m00"])

    # ---------- Kalman correction ----------
    if centro_detectado_small is not None:
        meas = np.array([[np.float32(centro_detectado_small[0])],
                         [np.float32(centro_detectado_small[1])]], dtype=np.float32)
        kalman.correct(meas)
        centro_k_small = centro_detectado_small
    else:
        centro_k_small = (cx_pred, cy_pred)

    # ---------- Estimar raio para detectar pinça ----------
    if iris_cont_small is not None:
        area = cv2.contourArea(iris_cont_small)
        r_est = int(np.sqrt(area/np.pi))
    else:
        r_est = int(min(img_small.shape[:2])*0.3)

    # ---------- Pinça ----------
    pinca_cont_small, mask_pinca = detectar_pinca_sobel_setor(
        img_small, centro_k_small, r_est
    )

    # ---------- Pintar ----------
    out = pintar_saida(img, iris_cont_small, pinca_cont_small, centro_k_small)
    return out



# ============================================================
# DISCO / PASTAS
# ============================================================

def listar_arquivos(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, f))]

def listar_subdirs(d):
    return [os.path.join(d, x) for x in sorted(os.listdir(d))
            if os.path.isdir(os.path.join(d, x))]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base = "data/preprocessed"
    pastas = listar_subdirs(base)

    for pasta in pastas:
        files = listar_arquivos(pasta)

        save_dir = os.path.join("data/preprocessed_dec", os.path.basename(pasta))
        os.makedirs(save_dir, exist_ok=True)

        for f in files:
            out = processar_frame(f)
            if out is not None:
                cv2.imwrite(os.path.join(save_dir, os.path.basename(f)), out)
                print("Salvo:", f)
            else:
                print("ERRO:", f)
