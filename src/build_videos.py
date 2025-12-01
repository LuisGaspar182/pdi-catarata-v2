import cv2
import os
from natsort import natsorted

def frames_to_video(frames_dir, output_path, fps=30):
    """Converte todos os frames de um diretório em um vídeo .mp4"""

    frames = [f for f in os.listdir(frames_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not frames:
        print(f"[AVISO] Nenhum frame encontrado em {frames_dir}")
        return

    frames = natsorted(frames)

    first_frame_path = os.path.join(frames_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"[ERRO] Não conseguiu ler: {first_frame_path}")
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Criando vídeo {output_path} ...")

    for f in frames:
        path = os.path.join(frames_dir, f)
        img = cv2.imread(path)

        if img is None:
            print(f"[ERRO] Frame inválido: {path}")
            continue

        writer.write(img)

    writer.release()
    print(f"[OK] Vídeo salvo: {output_path}")


def process_all_subdirs(base_dir="data/preprocessed_dec", output_dir="data/videos_processados"):
    """Percorre todos os subdirs de preprocessed_dec e cria vídeos"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sub in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        output_path = os.path.join(output_dir, f"{sub}.mp4")

        frames_to_video(sub_path, output_path, fps=30)


if __name__ == "__main__":
    process_all_subdirs()
