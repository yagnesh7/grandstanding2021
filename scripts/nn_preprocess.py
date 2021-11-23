import multiprocessing

import numpy as np
import pandas as pd
import librosa
import feature_extract as fe
from time import time
import os.path as osp


def extract(
    file,
    metadata,
    wav_dir,
    output_dir,
    target_sr=16000,
    target_channel=1,
    pitch_range=("C2", "C5"),
    log_pitch = True
):
    """worker function"""
    print(file)
    file_transcripts = metadata.loc[metadata["file"] == file].reset_index(drop=True)
    print(f"{file_transcripts.shape[0]} viable lines.")

    file_path = osp.join(wav_dir, file)
    # Load in audio
    audio_array, audio_sr = librosa.load(
        file_path, sr=librosa.core.get_samplerate(file_path)
    )

    # Pre-process (resample, rechannel)
    audio_array = fe.resample(audio_array, audio_sr, target_sr)
    audio_array = fe.rechannel(audio_array, target_channel)

    st = time()
    shapes = []
    for x in file_transcripts.iterrows():
        row = x[1]
        print(
            "File",
            file,
            "Item:",
            x[0],
            " |  Line:",
            row["line"],
            " |  Progress:",
            f"{round(100*(x[0]/file_transcripts.shape[0]))}%",
        )
        print(f"Elapsed Time: {round(time() - st, 2)}s")

        segment = audio_array[row["start_idx"] : row["end_idx"]]


        f0_inputed, voiced_flags, voiced_probs = fe.extract_pitch(
            segment, pitch_range, log=log_pitch
        )
        onset_strengths, onset_flags = fe.extract_onset(segment, target_sr)
        full_array = np.column_stack(
            (f0_inputed, voiced_flags, voiced_probs, onset_strengths, onset_flags)
        )
        np.save(osp.join(output_dir, f"{row['file']}_{row['line']}.npy"), full_array)

        shapes.append(full_array.shape)

    # For record keeping, used to decide max_lens
    np.save(
        osp.join(output_dir, f"{file_path.name}_shapes.npy"),
        np.array([s[0] for s in shapes]),
    )
    print(f"Total Time: {round(time() - st, 2)}s")

    return


if __name__ == "__main__":

    transcripts = pd.read_csv("../outputs/all_transcripts.csv")
    file_names = transcripts["file"].unique()
    target_sr = 16000
    target_channel = 1
    pitch_range = ("C2", "C5")
    log_pitch = True

    wav_dir = "../wavs/"
    write_dir = "../outputs/npy/"

    jobs = []
    for i, f in enumerate(file_names):
        # extract(f, transcripts, wav_dir, write_dir, target_sr, target_channel, pitch_range, log_pitch)
        p = multiprocessing.Pool(
            processes=multiprocessing.cpu_count()-1,
            target=extract,
            args=(f, transcripts, wav_dir, write_dir, target_sr, target_channel, pitch_range, log_pitch),
        )
        jobs.append(p)
        p.start()
