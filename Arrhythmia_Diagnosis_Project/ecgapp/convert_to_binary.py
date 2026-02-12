import os
import shutil

# Original multi-class dataset
source_base = "dataset/ecg_img"

# New binary-class dataset
target_base = "dataset/ecg_binary"

# Class definitions
normal_classes = ['n']                     # Normal class
abnormal_classes = ['f', 'v', 's', 'q']    # All other classes considered abnormal

# Helper function to copy files to binary folder structure
def copy_images(split):
    source_path = os.path.join(source_base, split)
    target_normal = os.path.join(target_base, split, 'normal')
    target_abnormal = os.path.join(target_base, split, 'abnormal')

    os.makedirs(target_normal, exist_ok=True)
    os.makedirs(target_abnormal, exist_ok=True)

    # Copy normal images
    for cls in normal_classes:
        cls_path = os.path.join(source_path, cls)
        if os.path.exists(cls_path):
            for fname in os.listdir(cls_path):
                src = os.path.join(cls_path, fname)
                dst = os.path.join(target_normal, fname)
                shutil.copy(src, dst)

    # Copy abnormal images
    for cls in abnormal_classes:
        cls_path = os.path.join(source_path, cls)
        if os.path.exists(cls_path):
            for fname in os.listdir(cls_path):
                src = os.path.join(cls_path, fname)
                dst = os.path.join(target_abnormal, f"{cls}_{fname}")
                shutil.copy(src, dst)

# Run for both train and test
copy_images('train')
copy_images('test')

print("✅ Binary dataset created at:", target_base)
