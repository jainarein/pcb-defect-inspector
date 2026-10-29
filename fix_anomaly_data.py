import os
import shutil

good_dir = "inhouse_boards/good"

# Add DeepPCB template images (these are defect-free reference boards)
pcb_root = "DeepPCB/PCBData"
count = 0
for group in os.listdir(pcb_root):
    group_path = os.path.join(pcb_root, group)
    if not os.path.isdir(group_path):
        continue
    for sub in os.listdir(group_path):
        sub_path = os.path.join(group_path, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in os.listdir(sub_path):
            if f.endswith("_temp.jpg"):
                shutil.copy2(os.path.join(sub_path, f), f"{good_dir}/deep_{group}_{f}")
                count += 1

good_total = len(os.listdir(good_dir))
bad_total = len(os.listdir("inhouse_boards/defective"))
print(f"Added {count} DeepPCB templates")
print(f"Good: {good_total}, Defective: {bad_total}")
