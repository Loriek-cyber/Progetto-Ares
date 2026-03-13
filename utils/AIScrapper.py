import struct
import matplotlib.pyplot as plt
import numpy as np


def plot_ac_fast_lane(file_path):
    points = []

    try:
        with open(file_path, "rb") as f:
            # Il primo intero (4 byte) è il numero di punti nella spline
            header = f.read(4)
            if not header: return
            num_points = struct.unpack('i', header)[0]
            print(f"Punti trovati nel file: {num_points}")

            for _ in range(num_points):
                # Ogni punto è composto da 7 float (4 byte ciascuno = 28 byte)
                # [x, y, z, track_pos, grip, left_width, right_width]
                data = f.read(28)
                if len(data) < 28: break

                # 'f' sta per float, ne leggiamo 7
                p = struct.unpack('fffffff', data)
                points.append(p)
    except FileNotFoundError:
        print("Errore: File non trovato. Controlla il percorso!")
        return

    # Convertiamo in array numpy per manipolare i dati facilmente
    points = np.array(points)

    # In AC, X e Z sono le coordinate del piano terra, Y è l'altezza
    x = points[:, 0]
    z = points[:, 2]
    velocity_proxy = points[:, 3]  # Spesso usato per indicare il progresso o grip

    # Creazione del Grafico
    plt.figure(figsize=(12, 8))

    # Usiamo un colormap (es. 'jet') per rendere la linea "viva"
    scatter = plt.scatter(x, z, c=range(len(x)), cmap='viridis', s=1)
    plt.colorbar(scatter, label='Progresso sul giro (Index)')

    plt.title(f"Mappa della Traiettoria Ideale - {num_points} punti")
    plt.xlabel("Coordinata X")
    plt.ylabel("Coordinata Z (Profondità)")
    plt.axis('equal')  # Mantiene le proporzioni reali della pista
    plt.grid(True, linestyle='--', alpha=0.6)

    print("Generazione mappa in corso...")
    plt.show()


path = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks/monza/ai/fast_lane.ai"
plot_ac_fast_lane(path)
