import os
import json
import random


random.seed(42)


img_dirs = [
    "training_data/unlabeled_frames_02_05",
    "training_data/unlabeled_frames_02_07"
]

imgs = []

for img_dir in img_dirs:
    imgs += [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
    ]

imgs = sorted(imgs)

print("Total images:", len(imgs))


assert len(imgs) == 200, f"Expected 200 images, found {len(imgs)}"

random.shuffle(imgs)

random.shuffle(imgs)

train_list = imgs[:120]   
val_list   = imgs[120:160]
test_list  = imgs[160:]

with open("train.json", "w") as f:
    json.dump(train_list, f, indent=2)

with open("val.json", "w") as f:
    json.dump(val_list, f, indent=2)

with open("test.json", "w") as f:
    json.dump(test_list, f, indent=2)

print(len(train_list), len(val_list), len(test_list))