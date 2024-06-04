from datasets import Dataset, Audio
from argparse import ArgumentParser
from moviepy.editor import *
import os
import glob
import xmltodict
import json

AUDIO_FOLDER = "data"

def mp4_to_mp3(mp4: str, mp3: str) -> None:
    mp4_without_frames = AudioFileClip(mp4)
    mp4_without_frames.write_audiofile(mp3)
    mp4_without_frames.close()

def mp4_to_wav(mp4: str, wav: str) -> None:
    """Convert an mp4 audio file to wav."""
    assert mp4.endswith(".mp4")
    assert wav.endswith(".wav")
    sound = AudioFileClip(mp4)
    sound.write_audiofile(wav)

def process_elan(folders: list) -> dict:
    audio_trans_list = list()
    cwd = os.getcwd()
    for folder in folders:
        i = 1
        audio_trans = dict()

        folder = os.path.basename(folder)
        assert os.path.exists(f"{AUDIO_FOLDER}/{folder}/{str(i)}")
        while os.path.exists(f"{AUDIO_FOLDER}/{folder}/{str(i)}"):
            # Annotation
            eafpath = f"{AUDIO_FOLDER}/{folder}/{str(i)}/{str(i)}.eaf"
            eafpath = glob.glob(eafpath)[-1]
            with open(eafpath, "r") as f:
                xml = f.read()
            dct = xmltodict.parse(xml)
            trans = dct["ANNOTATION_DOCUMENT"]["TIER"]["ANNOTATION"]["ALIGNABLE_ANNOTATION"]["ANNOTATION_VALUE"]

            # Audio
            audiopath = f"{AUDIO_FOLDER}/{folder}/{str(i)}/*.wav"
            if len(glob.glob(audiopath)) >= 1:
                wavpath = glob.glob(audiopath)[-1]
            else:
                mp4path = audiopath[:-3] + "mp4"
                mp4path = glob.glob(mp4path)[-1]
                wavpath = mp4path[:-3] + "wav"
                mp4_to_wav(mp4path, wavpath)
            
            print(wavpath, trans)
            audio_trans[str(i)] = {"audio_path": wavpath, "transcription": trans}
            i += 1
        audio_trans_list.append(audio_trans)
    return audio_trans_list

def filter_short_audio(batch):
    """Filter out audio samples that are shorter than 1 sec.
    If an audio sample is too short, it might cause `RuntimeError:
    Calculated padded input size per channel: (1). Kernel size: (3).
    Kernel size can't be greater than actual input size.
    """
    sr = batch["audio"]["sampling_rate"]
    return 1 < (len(batch["audio"]["array"]) / sr)

def process_audio(audio_trans_list: list, filter_short=True) -> Dataset:
    audio_list = []
    trans_list = []
    for audio_trans in audio_trans_list:
        for v in audio_trans.values():
            audio_list.append(v["audio_path"])
            trans_list.append(v["transcription"])

    audio_dataset = Dataset.from_dict({"audio": audio_list}).cast_column("audio",
                                                                         Audio(sampling_rate=16000))
    audio_dataset = audio_dataset.add_column("sentence", trans_list)
    if filter_short:
        audio_dataset = audio_dataset.filter(filter_short_audio, num_proc=4)
    return audio_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str,
                        default="KichwaAudio.json",
                        help="Dataset name of Kichwa audio with transcript.")
    args = parser.parse_args()

    audio_trans = process_elan()
    print("ELAN files processed")
    audio_dataset = process_audio(audio_trans)
    print("Audio files processed")
    audio_dataset.to_json(args.output)
    print("Dataset created")
    
