import os

train_dir = "I:/algae_data/train"
with open("classes.txt", "w") as f:
    for folder in sorted(os.listdir(train_dir)):
        f.write(folder + "\n")