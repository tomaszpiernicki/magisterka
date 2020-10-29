import glob
import os

from audio_feat_gen import data_gen_utls
from audio_feat_gen import feature

import torch
import numpy as np

from audio_feat_gen.data_gen_utls import load_desc_file
from run_training import get_model
from trainer import get_dataloaders

import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)(?!.*\d)',text)]

my_list =['Hello1', 'Hello12', 'Hello29', 'Hello2', 'Hello17', 'Hello25']
my_list.sort(key=natural_keys)

midi_to_abc_violin = {
    "40": "E,,",
    "41": "F,,",
    "42": "^F,,",
    "43": "G,,",
    "44": "^G,,",
    "45": "A,,",
    "46": "^A,,",
    "47": "B,,",

    "48": "C,",
    "49": "^C,",
    "50": "D,",
    "51": "^D,",
    "52": "E,",
    "53": "F,",
    "54": "^F,",
    "55": "G,",
    "56": "^G,",
    "57": "A,",
    "58": "^A,",
    "59": "B,",

    "60": "C",  # middle c
    "61": "^C",
    "62": "D",
    "63": "^D",
    "64": "E",
    "65": "F",
    "66": "^F",
    "67": "G",
    "68": "^G",
    "69": "A",
    "70": "^A",
    "71": "B",

    "72": "c",
    "73": "^c",
    "74": "d",
    "75": "^d",
    "76": "e",
    "77": "f",
    "78": "^f",
    "79": "g",
    "80": "^g",
    "81": "a",
    "82": "^a",
    "83": "b",

    "84": "c'"
}

def make_class_labels(midi_range):
    min_midi = min(midi_range)
    max_midi = max(midi_range)

    class_labels = {}
    for i in range(max_midi - min_midi + 1):
        class_labels[f"{min_midi + i}"] = i
    return class_labels

def get_midi_lenght(midi, midi_array):
    midi_len = 0
    for idx, midis in enumerate(midi_array):
        if midi in midis:
            midi_len += 1
            midi_array[idx].remove(midi)
        else:
            break
    return midi_len

def resolve_note_lenght(value):
    if value == 1:  # ósemka
        return ""
    elif value == 2:  # ćwiercnuta
        return "2"
    elif value == 3:  # ćwiercnuta >
        return "2>"
    elif value == 4 or value == 5:  # półnuta
        return "4"
    elif value == 6 or value == 7:  # półnuta >
        return "4>"
    elif value == 8 or value == 9:  # całanuta
        return "8"
    else:
        return "8>"  # całanuta >


def create_abc_notation(id, title, metrum, tempo, midi_anwsers, outfile_name, L = "1/8"):
    file_handle = open(outfile_name, "a")

    header_lines = []
    header_lines.append(f"X: {id} \n")
    header_lines.append(f"T: {title} \n")
    header_lines.append(f"Q: {tempo} \n")
    header_lines.append(f"M: {metrum} \n")
    header_lines.append(f"L: {L}\n")
    header_lines.append(f"K: C clef=treble\n")

    for line in header_lines:
        file_handle.write(line)

    midi_lengths = []
    start_idx = 0
    for start_idx, midis in enumerate(midi_anwsers):
        if not midis:
            continue
        midi_lengths.append({})
        for midi in midis:
            midi_length = get_midi_lenght(midi, midi_anwsers[start_idx:])
            midi_lengths[-1][midi] = midi_length

    abc_notes = ""
    for midi_len in midi_lengths:
        abc_entry = "["
        for key, value in midi_len.items():
            note_abc = midi_to_abc_violin[key]
            abc_entry += note_abc
            note_abc_len = resolve_note_lenght(value)
            abc_entry += note_abc_len
            abc_entry += " "
        abc_entry += "] "
        abc_notes += (abc_entry)

    file_handle.write(abc_notes)

    file_handle.close()


temp_folder = "e:/temporary"
wav_file = f"{temp_folder}/data_fold_0.wav"
test_name = "prymy"
temp_sub_folder = f"{temp_folder}/{test_name}"
# wav_file = f"e:\\Dataset\\0.6\\{test_name}\\audio\\data_fold_0.wav"
# wav_file = "e:/temporary/pachelbel.wav"
wav_file = "E:\\temporary\\pachelbel-testowy\\prymy.wav"
desc_file = f"e:\\Dataset\\0.6\\{test_name}\\meta\\meta_fold_0.csv"
feat_folder = f"{temp_sub_folder}/features"

os.makedirs(feat_folder, exist_ok=True)

outfile_name = f"{temp_sub_folder}/outfile.abc"
orignal_name = f"{temp_sub_folder}/original.abc"

tmp_feat_file = feature.extract_features_single_file(wav_file, temp_folder, is_mono = True, nfft=1024, nb_mel_bands = 40, feat_mode="spec")
x = feature.normalize_sigle_file(tmp_feat_file)
data_gen_utls.save_in_parts_signle(x, 16, temp_sub_folder, 0)

checkpoint_path = 'E:\\checkpoints\\resnet_mix_v6\\checkpoint_epoch585.pth'
# checkpoint_path = 'E:\\checkpoints\\resnet_single_v6\\checkpoint_epoch1337.pth'

checkpoint = torch.load(checkpoint_path)

device = "cuda"
class_labels = make_class_labels([40, 84])

model = get_model("OneChannelResnet", device, class_labels)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

folds = 1
batch_size = 32

feat_files = sorted(glob.glob(feat_folder + '/mbe_*_fold0_*.npz'), key=os.path.getmtime)   # Fold 0.

anwsers = []
for file in feat_files:
    dmp = np.load(file, allow_pickle=True)
    _X_train = dmp['arr_0']
    inputs = torch.from_numpy(_X_train)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.to(device='cuda')
    outputs = model(inputs)
    outputs = torch.round(outputs.to('cpu').detach()).numpy()
    outputs = outputs[0]
    indexes = np.where(outputs == 1)[0]
    anwsers.append(indexes)

midi_anwsers = []
for anwser in anwsers:
    midi_anwser = []
    for indicies in anwser:
        midi_anwser.append(list(class_labels.keys())[list(class_labels.values()).index(indicies)])
    midi_anwsers.append(midi_anwser)

# desc_list = load_desc_file(desc_file, class_labels, map_midis=False)
#
# real_time_len = 0.0
# for l in desc_list[list(desc_list.keys())[0]]:
#     if real_time_len < float(l[1]):
#         real_time_len = float(l[1])
#
# time_resulution = 60 / len(feat_files)  # 0.5 sekundy
# original_list_len = round(real_time_len / time_resulution)
#
# original_midi_anwsers = [None] * original_list_len
#
# for idx in range(len(original_midi_anwsers)):
#     original_midi_anwsers[idx] = []
#
# for lst in desc_list[list(desc_list.keys())[0]]:
#     midi_key = lst[-1]
#     start_time = lst[0]
#     end_time = lst[1]
#
#     start_idx = round(start_time / time_resulution) - 1
#     delta_time = end_time - start_time
#     if delta_time < time_resulution:
#         idx_len = 1
#     else:
#         idx_len = round(delta_time / time_resulution)
#     for i in range(idx_len):
#         original_midi_anwsers[start_idx + i].append(str(midi_key))

# create_abc_notation("1", "Original", "4/4", "60", original_midi_anwsers, orignal_name)
create_abc_notation("2", "Test", "4/4", "60", midi_anwsers, outfile_name)
