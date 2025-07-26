import shutil
import random
import os

def copy_subset_per_class(src_dir, dst_dir, max_per_class=100, exts=('.jpg', '.jpeg', '.png')):
    """
    Copies up to max_per_class images from each subfolder in src_dir to dst_dir, preserving folder structure.
    If a subfolder has fewer than max_per_class images, copies all images.

    Parameters:
    ----------
        src_dir (str): Source directory with class subfolders.
        dst_dir (str): Destination directory to copy subset.
        max_per_class (int): Maximum images to copy per class.
        exts (tuple): Allowed image file extensions.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for class_name in os.listdir(src_dir):
        class_src = os.path.join(src_dir, class_name)
        class_dst = os.path.join(dst_dir, class_name)

        # Skip if it's not a directory (e.g., a file)
        if not os.path.isdir(class_src):
            continue
        
        os.makedirs(class_dst, exist_ok=True)
        # List image files
        images = [f for f in os.listdir(class_src) if f.lower().endswith(exts)]
        random.shuffle(images)
        subset = images[:max_per_class]
        for img in subset:
            src_path = os.path.join(class_src, img)
            dst_path = os.path.join(class_dst, img)
            shutil.copy2(src_path, dst_path)