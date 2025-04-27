import os
import argparse
import numpy as np
import demucs.separate
import shlex
import logging
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

DB_PATH = "/home/jacob/Desktop/Projects/demucs/datasets/musdb18hq/"
SEPARATED_PATH = "/home/jacob/Desktop/Projects/demucs/tools/separated/htdemucs"

np.set_printoptions(formatter={'float': '{:0.3f}'.format})

N_CLASSIFICATIONS = 4
DRUMS = 0
BASS = 1
OTHER = 2
VOCALS = 3
def generate_stems(model_name):
    test_path = os.path.join(DB_PATH, "test")
    for dirname in os.listdir(test_path):
        dir_path = os.path.join(test_path, dirname)
        mixture_path = os.path.join(dir_path,"mixture.wav") 
        final_path = os.path.join(dir_path, f"{dirname}.wav")
        try:
            os.rename(mixture_path, os.path.join(dir_path, f"{dirname}.wav"))
            
        except FileNotFoundError as e:
            logging.debug("mixture.wav not found. Skipping rename...")

        demucs.separate.main(shlex.split(f'-n {model_name} "{final_path}"'))

def get_separated_stft(separated_path, window_size=1024, hop_size=512):

    target_stems = {}
    for dirname in os.listdir(separated_path):
        base_path = os.path.join(separated_path, dirname)
        # These are tuples with [0] == sample rate, [1] == data
        target_stems["drums"] = wavfile.read(os.path.join(base_path, "drums.wav"))
        target_stems["bass"] = wavfile.read(os.path.join(base_path, "bass.wav"))
        target_stems["other"] = wavfile.read(os.path.join(base_path, "other.wav"))
        target_stems["vocals"] = wavfile.read(os.path.join(base_path, "vocals.wav"))

    dataset_stems = {}
    for dirname in os.listdir(os.path.join(DB_PATH, "test")):
        base_path = os.path.join(DB_PATH, "test", dirname)
        dataset_stems["drums"] = wavfile.read(os.path.join(base_path, "drums.wav"))
        dataset_stems["bass"] = wavfile.read(os.path.join(base_path, "bass.wav"))
        dataset_stems["other"] = wavfile.read(os.path.join(base_path, "other.wav"))
        dataset_stems["vocals"] = wavfile.read(os.path.join(base_path, "vocals.wav"))

    # Assume all stems have same sampling rate
    SFT = ShortTimeFFT(np.hamming(window_size), hop_size, fs=float(target_stems["drums"][0]))
    target_fstems = {}
    dataset_fstems = {}
    for key in target_stems:
        target_fstems[key] = SFT.stft(target_stems[key][1].T)
        dataset_fstems[key] = SFT.stft(dataset_stems[key][1].T)

    confusion_matrix = np.zeros((4,4))
    indices = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}

    # For each timestep, we check the predicted frequencies in each stem
    # Example: We check the generated drums stem
    # Then we compare that stem to the four real data stems
    # The smallest distance between the generated and the real data is determined to be what the target was 'classified as'
    # E.g. when checking the drums stem, if we are closest to the actual 'bass' stem, then we consider that a misclassification
    # where the 'actual' value was drums, but we 'classified it' as bass
    for i in range(target_fstems[key].shape[2]):
        
        for key in target_fstems:
            target = abs(target_fstems[key][:, :, i])
            dist = {}
            for data_key in dataset_fstems:
                data = abs(dataset_fstems[data_key][:, :, i])
                dist[data_key] = np.linalg.norm((data - target), ord=2)

            prediction = min(dist, key=dist.get)
            #print(f"EXPECTED: {prediction}, ACTUAL: {key}")
            confusion_matrix[indices[key]][indices[prediction]] += 1

    print(confusion_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix, 
                                  display_labels=np.array(["drums","bass","other","vocals"]))
    disp.plot()
    plt.show()


def main(args):
    """
    if args.model:
        generate_stems(args.model)
    else:
        generate_stems("htdemucs")
    """

    get_separated_stft(SEPARATED_PATH)


    

        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model name")
    args = parser.parse_args()
    main(args)  