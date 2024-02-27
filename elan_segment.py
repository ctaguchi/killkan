import xmltodict
from argparse import ArgumentParser
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from datetime import datetime
import os, glob
from xml.dom.minidom import parseString
from typing import Union, List, Tuple
import pandas as pd
import tqdm

def clip_mp4(src: str,
             start: str,
             end: str,
             idx: str,
             trg_dir: str) -> dict:
    """Segment the original mp4 into multiple mp4 files.

    Params:
    src: Source mp4 file
    start: Start time of the segment
    end: End time of the segment
    trg_dir: Target directory, e.g., "1".

    Returns:
    dict: A dict of the information of the segmented audio for
    creating the output .eaf file.
    """
    # divide by 1000 to adjust the input format to moviepy
    if not os.path.exists(trg_dir):
        os.mkdir(trg_dir)
    trg = f"{trg_dir}/{idx}.mp4"
    # trg = trg_dir + "/" + src[:-4].split("/")[-1] + "{}.mp4".format(idx)
    ffmpeg_extract_subclip(src,
                           int(start) / 1000,
                           int(end) / 1000,
                           targetname=trg)
    trg_abspath = "file://" + os.path.abspath(trg)
    trg_relpath = os.path.relpath(trg)
    output = {"src": src,
              "start": start,
              "end": end,
              "abspath": trg_abspath,
              "relpath": trg_relpath}
    return output

def eaf2segments(eaf_file: str) -> tuple:
    """
    Segment the original eaf file into multiple eaf files per annotation.

    Params:
    eaf_file: Path to the eaf original file.

    Returns:
    tuple: A tuple of a list of annotations and a list of timestamps.
    """
    with open(eaf_file) as f:
        xml = f.read()
    dct = xmltodict.parse(xml)
    annotations = dct["ANNOTATION_DOCUMENT"]["TIER"]["ANNOTATION"]
    timestamps = dct["ANNOTATION_DOCUMENT"]["TIME_ORDER"]["TIME_SLOT"]
    return dct, annotations, timestamps

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", nargs="+", default=None,
                        help="Specify the directory name that you want to parse." \
                        "You can specify multiple directories.")
    parser.add_argument("-e", "--eaf", default=None,
                        help="Input original .eaf file.")
    parser.add_argument("-a", "--audio", default=None,
                        help="Input original .mp4 file.")
    parser.add_argument("-ae", "--audio_extension", default=".mp4",
                        help="Specify the extension of the audio files.")
    parser.add_argument("-D", "--debug", action='store_true',
                        help="Debug mode if specified")
    args = parser.parse_args()
    return args

class ELANtoUD():
    """
    A class of functions to convert ELAN eaf files into UD-style conllu file.
    Thsis class is under construction.
    """
    def __init__(self, ud=True):
        self.ud = ud

    def eaftodict(self, eaf_file: str) -> dict:
        """Convert the EAF file into a Python dict."""
        with open(eaf_file) as f:
            xml = f.read()
        dct = xmltodict.parse(xml)
        return dct

    def get_tier_list(self, dct: dict) -> list:
        """Get a list of tiers in the annotation."""
        tier_list = dct["ANNOTATION_DOCUMENT"]["TIER"]
        if not type(tier_list) == list: # If the annotation contains only one tier
            tier_list = [tier_list]
        return tier_list

    def get_tier_content(self, tier_list: list) -> dict:
        """Get a list of dicts of annotation content.

        Each tier content is a dictionary that looks like this:

        {'@ANNOTATOR': 'Chihiro Taguchi',
        '@DEFAULT_LOCALE': 'en',
        '@LINGUISTIC_TYPE_REF': 'FORM',
        '@PARENT_REF': 'default',
        '@TIER_ID': 'FORM',
        'ANNOTATION': [{'REF_ANNOTATION': {'@ANNOTATION_ID': 'a13',
        '@ANNOTATION_REF': 'a10',
        'ANNOTATION_VALUE': 'Chaytaka'}}, ... ]

        If there is no annotated content in the tier, the dictionary
        does not have the "ANNOTATION" key.
        """
        
        transcription_tier = tier_list[0] # default tier
        assert transcription_tier["@TIER_ID"] == "default"

        if True: # We do not touch other ties yet.
            ud_tiers = {"text": transcription_tier}
            return ud_tiers
        
        form_tier = tier_list[1]
        assert form_tier["@TIER_ID"] == "FORM"

        spanish_tier = tier_list[2]
        assert spanish_tier["@TIER_ID"] == "Spanish"

        lemma_tier = tier_list[3]
        assert lemma_tier["@TIER_ID"] == "LEMMA"

        upos_tier = tier_list[4]
        assert upos_tier["@TIER_ID"] == "UPOS"

        feats_tier = tier_list[5]
        assert feats_tier["@TIER_ID"] == "FEATS"

        id_tier = tier_list[6] # Somehow ID's position is messed up
        assert id_tier["@TIER_ID"] == "ID"

        head_tier = tier_list[7]
        assert head_tier["@TIER_ID"] == "HEAD"

        deprel_tier = tier_list[8]
        assert deprel_tier["@TIER_ID"] == "DEPREL"

        misc_tier = tier_list[9]
        assert misc_tier["@TIER_ID"] == "MISC"

        notes_tier = tier_list[10] # Not for UD.
        assert notes_tier["@TIER_ID"] == "Notes"

        ud_tiers = {"text": transcription_tier,
                    "FORM": form_tier,
                    "Spanish": spanish_tier,
                    "LEMMA": lemma_tier,
                    "UPOS": upos_tier,
                    "FEATS": feats_tier,
                    "ID": id_tier,
                    "HEAD": head_tier,
                    "DEPREL": deprel_tier,
                    "MISC": misc_tier}
        return ud_tiers

    def get_text_and_timestamps(self, dct: Union[dict, str], align=True):
        """Get tuples of text and its corresponding start/end timestamps.
        
        args:
        - dct: The dict converted from original EAF file, or the path (str)
        to the original EAF file.
        - align: If True, this function will give the time-aligned annotation
        that looks like this:
        [{"annotation_id": "a1",
        "text": "lorem ipsum",
        "start": "2940",
        "end": "6180"}, 
        ... ]
        """
        if type(dct) == str:
            dct = self.eaftodict(dct)
        ud_tiers = self.get_tier_content(self.get_tier_list(dct))
        transcription_tier = ud_tiers["text"]["ANNOTATION"] # -> a list of dicts
        timestamps = dct["ANNOTATION_DOCUMENT"]["TIME_ORDER"]["TIME_SLOT"] # -> a list of timestamps
        
        if align:
            assert len(timestamps) == len(transcription_tier) * 2
            alignments_list = list()
            for i, content in enumerate(transcription_tier):
                """Each `content` looks like this:
                {'ALIGNABLE_ANNOTATION': {'@ANNOTATION_ID': 'a27',
                '@TIME_SLOT_REF1': 'ts29',
                '@TIME_SLOT_REF2': 'ts30',
                'ANNOTATION_VALUE': 'Kacharipay, pushak alcaldesa!'}}
                """
                alignments = dict()
                stamp = i * 2
                start = timestamps[stamp]["@TIME_VALUE"]
                end = timestamps[stamp+1]["@TIME_VALUE"]
                text = content["ALIGNABLE_ANNOTATION"]["ANNOTATION_VALUE"]
                annotation_id = content["ALIGNABLE_ANNOTATION"]["@ANNOTATION_ID"]

                # test
                start_timeslot_id = content["ALIGNABLE_ANNOTATION"]["@TIME_SLOT_REF1"]
                end_timeslot_id = content["ALIGNABLE_ANNOTATION"]["@TIME_SLOT_REF2"]
                assert start_timeslot_id == timestamps[stamp]["@TIME_SLOT_ID"] or \
                    start_timeslot_id == timestamps[stamp-1]["@TIME_SLOT_ID"], print(start_timeslot_id, timestamps[stamp]["@TIME_SLOT_ID"])
                assert end_timeslot_id == timestamps[stamp+1]["@TIME_SLOT_ID"] or \
                    end_timeslot_id == timestamps[stamp+2]["@TIME_SLOT_ID"], print(end_timeslot_id, timestamps[stamp+1]["@TIME_SLOT_ID"])
                
                alignments["text"] = text
                alignments["start"] = start
                alignments["end"] = end
                alignments["annotation_id"] = annotation_id
                alignments_list.append(alignments)
            return alignments_list
        else:
            return transcription_tier, timestamps

    def get_text_and_translation(self, dct: dict) -> List[Tuple]:
        ud_tiers = self.get_tier_content(self.get_tier_list(dct))
        print([name for name in ud_tiers.keys()])
        transcription_tier = ud_tiers["text"]["ANNOTATION"] # -> a list of dicts
        translation_tier = ud_tiers["Spanish"]["ANNOTATION"] # -> a list of dicts

        transcriptions = [entry["ALIGNABLE_ANNOTATION"]["ANNOTATION_VALUE"]
                          for entry in transcription_tier]
        translations = [entry["REF_ANNOTATION"]["ANNOTATION_VALUE"]
                        for entry in translation_tier]
        kichwa_spanish = list(zip(transcriptions, translations))
        return kichwa_spanish

    def get_kichwa_spanish_csv(self, kichwa_spanish: List[Tuple],
                               dst="kichwa_spanish.csv") -> None:
        """
        Create a csv file containing the aligned Kichwa sentences
        and their Spanish translations.

        Arguments:
        - `kichwa_spanish`: a list of tuples containing a Kichwa
        sentence and its Spanish translation.
        - `dst` (str): the path to the ouptut csv file for the
        dataframe to be saved.
        """
        df = pd.DataFrame(kichwa_spanish, columns=["Kichwa", "Spanish"])

        # Copy the Spanish column to keep track of the change.
        df["Original"] = df["Spanish"]
        df.to_csv(dst)

def get_all_chapters() -> list:
    """Get a list of dicts of eaf/audio file paths."""
    paths = []
    for i in range(1, 21):
        entry = dict()
        folder_name = "Chapter{}".format(i)
        eaf_name = "{} MASTER K{}.eaf".format(i, i)
        audio_name = "{} MASTER K{}.mp4".format(i, i)

        eaf_relpath = folder_name + eaf_name
        audio_relpath = folder_name + audio_name
        entry["eaf"] = eaf_relpath
        entry["audio"] = audio_relpath
        paths.append(entry)
    return paths

def write_eaf_file(alignments_list: list, chapter_dir: str):
    for i in range(len(alignments_list)):
        start = alignments_list[i]["start"]
        end = alignments_list[i]["end"]
        target_dir = f"data/{chapter_dir}/{str(i+1)}"
        output = clip_mp4(audio,
                          start,
                          end,
                          str(i+1),
                          trg_dir=target_dir)
        now = datetime.now()
        now = now.astimezone().isoformat("T", "seconds")
        
        annotation_id = alignments_list[i]["annotation_id"]
        text = alignments_list[i]["text"]
        eaf_text = EAF_TEXT.format(output["abspath"],
                                   output["relpath"],
                                   annotation_id,
                                   str(int(end) - int(start)),
                                   text)
        output_eaf = f"data/{chapter_dir}/{str(i+1)}/{str(i+1)}.eaf"
        with open(output_eaf, "w", encoding="utf-8") as f:
            f.write(eaf_text)

# `eaf_text` might be subject to change for the versions with UD-style tiers.
EAF_TEXT = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2023-06-05T23:42:28-05:00"
    FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR
            MEDIA_URL="{}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{}"/>
        <PROPERTY NAME="lastUsedAnnotationId">{}</PROPERTY>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="{}"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="default-lt" TIER_ID="default">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
                TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>{}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false"
        LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true"/>
    <CONSTRAINT
        DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT
        DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT
        DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>
"""

if __name__ == "__main__":
    args = get_args()
    elan2ud = ELANtoUD()
    
    if args.directory == None:
        args.directory = ["Chapter" + str(i) for i in range(1, 21)]

    for d in tqdm.tqdm(args.directory):
        eaf = glob.glob(f"data/{d}/{d}.eaf")[-1]
        audio = glob.glob(f"data/{d}/*{args.audio_extension}")[-1]
        
        dct = elan2ud.eaftodict(eaf)
        alignments_list = elan2ud.get_text_and_timestamps(dct) # -> list of dict of text/start/end/id
        write_eaf_file(alignments_list, d)
