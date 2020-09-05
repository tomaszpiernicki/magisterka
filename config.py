# import utils
# import torch
#
# version = "0.2"
# chord_name = "single"
# experiment_name = "generate_single"
#
# dataset_home = f"E:/Dataset/chords_v{version}/{chord_name}/"
# mode = "train"
# out_dir = f"{dataset_home}{mode}"
#
# dry_data_path = "E:\\Dataset\\nsynth\\guitar\\audio\\*"
# # f"E:/Dataset/nsynth/dataset/{mode}/*/audio/*guitar*"
#
# is_mono = True
# nb_ch = 1 if is_mono else 2
#
# folds = 100
# fold_size = 16  # Giving around 0.5 second long samples
#
# threshold = 0.5
#
# evaluation_setup_folder = f'{out_dir}/evaluation_setup/'
# audio_folder = f'{out_dir}/audio/'
# feat_folder = f'{out_dir}/features/'
# chpt_folder = f'{out_dir}/checkpoints/'
#
# utils.create_folder(evaluation_setup_folder)
# utils.create_folder(audio_folder)
# utils.create_folder(feat_folder)
# utils.create_folder(chpt_folder)
#
# data_file_name = "data_fold"
# meta_file_name = "meta_fold"
#
# nfft = 1024
# win_len = nfft
# hop_len = win_len / 2
# nb_mel_bands = 40
# sr = 16000
#
# # TODO: Change midi range to 24 - 95 ( 6 octaves )
#
# # No
# min_midi = 40  # Lowes tone in my guitar
# max_midi = 84  # Highest tone in my guitar
#
# __class_labels = {}
#
# for i in range(max_midi - min_midi + 1):
#     __class_labels[f"{min_midi + i}"] = i
#
# class_labels = __class_labels
#
# epochs = 1000
# restart_from_checkpoint = True
# restart_checkpoint = 1419
#
# batch_size = 16
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # END