import os
import shutil
import random

random.seed(42)

out = "inhouse_boards"
os.makedirs(f"{out}/good", exist_ok=True)
os.makedirs(f"{out}/defective", exist_ok=True)

# Clean boards from PCB_USED
src_good = "PCB_DATASET/PCB_DATASET/PCB_USED"
for f in os.listdir(src_good):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy2(os.path.join(src_good, f), f"{out}/good/{f}")

# Defective boards from each category
src_img = "PCB_DATASET/PCB_DATASET/images"
count = 0
for cat in os.listdir(src_img):
    cat_path = os.path.join(src_img, cat)
    if not os.path.isdir(cat_path):
        continue
    files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    picked = random.sample(files, min(10, len(files)))
    for f in picked:
        shutil.copy2(os.path.join(cat_path, f), f"{out}/defective/{cat}_{f}")
        count += 1

good_count = len(os.listdir(f"{out}/good"))
bad_count = len(os.listdir(f"{out}/defective"))
print(f"Good: {good_count}, Defective: {bad_count}")
