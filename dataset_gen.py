import torch
import math
import os

import torchaudio

from helper import _get_sample

def add_noise(speech: torch.Tensor, noise: torch.Tensor, snr: int) -> torch.Tensor:
    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    if noise.shape[1] < speech.shape[1]:
        tiling_factor = math.ceil(speech.shape[1] / noise.shape[1])
        noise = noise.tile((1, tiling_factor))

    noise = noise[:, :speech.shape[1]]

    snr_factor = math.exp(snr / 10)
    scale = snr_factor * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech

def generate_dataset(speech_directory: str, noise_directory: str, noisy_speech_directory:str) -> None:
    SAMPLE_RATE = 16000
    SNR = 3

    speech_files = [os.path.join(speech_directory, x) for x in os.listdir(speech_directory)]
    noise_files = list(os.listdir(noise_directory))
    noise_file_paths = [os.path.join(noise_directory, x) for x in noise_files]
    noise_waveforms = [_get_sample(path, resample=SAMPLE_RATE)[0] for path in noise_file_paths]

    for sfl in speech_files:
        speech_waveform, _ = _get_sample(sfl, resample=SAMPLE_RATE)

        for i in range(len(noise_waveforms)):
            noisy_speech = add_noise(speech_waveform, noise_waveforms[i], SNR)
            output_filename = os.path.basename(sfl).split(".")[0] + "_" + noise_files[i]

            torchaudio.save(os.path.join(noisy_speech_directory, output_filename), noisy_speech, SAMPLE_RATE)

if __name__ == "__main__":
    speech_directory = "assets/speech"
    noise_directory = "assets/noise"
    noisy_speech_directory = "assets/noisy_speech"

    generate_dataset(speech_directory, noise_directory, noisy_speech_directory)
