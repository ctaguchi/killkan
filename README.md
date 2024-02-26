# Killkan
Killkan is the first ASR dataset for the Kichwa language.
This repository contains both the dataset and the code used to preprocess the data and to train the ASR model.

## Source
"Jaboneropak Ayllullaktapi" by Radialistas Apasionadas y Apasionados (https://radialistas.net/).

## License
CC-BY 4.0

## Code
- `elan_segment.py`: This program segments the whole ELAN file (.eaf) and the audio file into a file per sentence.
- `elan2ud.py`: This converts an ELAN file (.eaf) into the UD-style CoNLL-U format.
- `kc_killkan-ud-test.conllu`: The UD-style CoNLL-U file.