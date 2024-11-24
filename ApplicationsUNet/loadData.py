import os
import numpy as np
from skimage import io
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch

class ThoraxDataLoader:
    """
    Classe pour charger les données thoraciques et les organiser en un tableau 3D.
    """
    def __init__(self, base_dir: str):
        """
        Initialise la classe avec le répertoire contenant les échantillons.
        
        :param base_dir: Chemin du dossier de base contenant les échantillons.
        """
        self.base_dir = base_dir
        self.data = None

        # Créer le colormap personnalisé
        top = cm.get_cmap('viridis', 64)
        bottom = cm.get_cmap('plasma', 960)
        newcolors = np.vstack((top(np.linspace(0, 1, 64)),
                               bottom(np.linspace(1, 0, 960))))
        self.newcmp = ListedColormap(newcolors, name='MonteCarlo')

    def load_all_samples(self, type) -> np.ndarray:
        """
        Parcourt les dossiers des échantillons, charge les fichiers et les empile dans un tableau 3D.
        
        :return: Tableau numpy de dimensions (sample_number, 64, 64).
        """
        all_samples = []

        for sample_dir in os.listdir(self.base_dir):
            sample_path = os.path.join(self.base_dir, sample_dir)

            if os.path.isdir(sample_path):  # Vérifie si c'est un dossier
                sample_data = self._load_sample(sample_path, type)

                if sample_data is not None:
                    all_samples.append(sample_data)
                    print(f"{sample_dir} chargé avec succès.")

                else:
                    print(f"{sample_dir} contient des fichiers manquants.")

        # Empiler les données en un tableau numpy
        if all_samples:
            self.data = np.stack(all_samples, axis=0)
        else:
            self.data = np.array([])

        print(f"Total {len(all_samples)} échantillons chargés.")
        return self.data

    def _load_sample(self, sample_path: str, type) -> np.ndarray:
        """
        Charge les fichiers nécessaires pour un échantillon donné.
        
        :param sample_path: Chemin du répertoire d'un échantillon.
        :return: Tableau numpy 2D (64, 64) représentant l'échantillon ou None si incomplet.
        """
        norm = Normalize(vmin=0, vmax=1)
        newcmp = self.newcmp

        try:
            files = {
                "LS": os.path.join(sample_path, "low_edep.mhd"),
                "HS": os.path.join(sample_path, "high_edep.mhd"),
                "CT": os.path.join(sample_path, "ct.mhd"),
                "MaskCT": os.path.join(sample_path, "mask_ct.mhd")
            }

            # Vérifie si tous les fichiers existent
            if not all(os.path.isfile(path) for path in files.values()):
                return None

            # Charger uniquement le fichier nécessaire (par ex. "HS")
            sample_array = io.imread(files[type], plugin='simpleitk')
            if type == "LS" or type == "HS":
                sample_array = cm.ScalarMappable(norm=norm, cmap=newcmp).to_rgba(sample_array)[:, :, :3]

            # Vérifier les dimensions
            if (sample_array.shape == (64, 64) and type in ["MaskCT", "CT"]) or (sample_array.shape == (64, 64, 3) and type in ["LS", "HS"]):
                return sample_array
            else:
                print(f"Dimensions invalides dans {sample_path}: {sample_array.shape}")
                return None
        except Exception as e:
            print(f"Erreur lors du chargement de l'échantillon dans {sample_path}: {e}")
            return None
