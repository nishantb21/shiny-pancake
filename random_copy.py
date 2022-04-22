import os
import random
import shutil

clean_speech_pth = "/mnt/c/Users/nisha/Documents/DL Final Project/assets/datasets/LibriSpeech/train-clean-100"
output_pth = "assets/speech"
upper_folder = os.listdir(clean_speech_pth)

for fldr in upper_folder:
    base_path = os.path.join(clean_speech_pth, fldr)
    inner_fldr = os.listdir(base_path)[0]
    afls = os.listdir(os.path.join(base_path, inner_fldr))
    afls = [x for x in afls if ".txt" not in x]

    moveable_files = random.choices(afls, k=4)

    for fl in moveable_files:
        try:
            shutil.copy(os.path.join(base_path, inner_fldr, fl), os.path.join(output_pth, fl))

        except:
            pass