import numpy as np
import librosa


def resample(audio, current_sr, target_sr):
    if current_sr == target_sr:
        return audio
    else:
        new_audio = librosa.resample(audio, current_sr, target_sr)
        return new_audio


def rechannel(audio, target_channel):
    current_shape = audio.shape
    if target_channel == 1:
        if len(current_shape) == 1:
            return audio
        else:
            new_audio = librosa.to_mono(audio)
            return new_audio
    else:
        if len(current_shape) == 2:
            return audio
        else:
            new_audio = np.column_stack([audio, audio])
            return new_audio


def extract_pitch(audio, pitch_range=("C2", "C5"), impute_val=0.0, log=True):
    f0, voiced_flags, voiced_probs = librosa.pyin(
        y=audio,
        fmin=librosa.note_to_hz(pitch_range[0]),
        fmax=librosa.note_to_hz(pitch_range[1]),
    )
    if log:
        f0 = np.log2(f0)
    f0_inputed = np.nan_to_num(f0, nan=impute_val)
    return (f0_inputed, voiced_flags, voiced_probs)


def extract_onset(audio, sampling_rate, max_size=5):
    onset_strengths = librosa.onset.onset_strength(
        y=audio, sr=sampling_rate, max_size=max_size
    )
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sampling_rate, units="frames")
    onset_flags = np.zeros(onset_strengths.shape[0])
    onset_flags[onset_frames] = 1
    return (onset_strengths, onset_flags)
