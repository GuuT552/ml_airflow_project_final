import shutil
import os

def save_to_local_storage(source_path: str, target_dir: str = 'results/'):
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    shutil.copy(source_path, os.path.join(target_dir, filename))