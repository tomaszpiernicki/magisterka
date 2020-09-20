import glob
import json
import os
import re

import utils
import shutil
import torch

from audio_feat_gen.data_gen_utls import pack_paths


class Config():
    def parse_config(self, config_file):
        raise(NotImplementedError)

    def save_config(self):
        raise(NotImplementedError)

    def make_class_labels(self, midi_range):
        self.min_midi = min(midi_range)
        self.max_midi = max(midi_range)

        self.class_labels = {}
        for i in range(self.max_midi - self.min_midi + 1):
            self.class_labels[f"{self.min_midi + i}"] = i

    def make_directories(self, working_directory, version, chord_name):
        self.out_dir = f"{working_directory}/{version}/{chord_name}"
        self.meta_directory = f"{self.out_dir}/meta"
        self.audio_directory = f"{self.out_dir}/audio"
        self.feature_folder = f'{self.out_dir}/features/'
        # self.evaluation_setup_folder = f'{self.out_dir}/evaluation_setup/'

    def load_config(self, config_file):
        self.config_file = config_file
        print(f"Loading config from: {self.config_file}")

        with open(self.config_file) as json_file:
            config = json.load(json_file)

        return config

class GeneratorConfig(Config):
    def parse_config(self, config_file):
        config = super().load_config(config_file)

        self.sr = config["sr"]
        self.folds = config["folds"]
        self.version = config["version"]
        self.chord_name = config["chord_name"]
        self.working_directory = config["working_directory"]
        self.midi_range = config["midi_range"]

        self.dry_data_paths = config["dry_data_paths"]
        self.audio_filename = config["audio_filename"]
        self.meta_filename = config["meta_filename"]
        self.overlap_list = config["overlap_list"]
        self.overlap_prob= config["overlap_prob"]
        self.zero_padding = config["zero_padding"]

        self.make_directories(self.working_directory, self.version, self.chord_name)

        utils.create_folder(self.out_dir)
        utils.create_folder(self.meta_directory)
        utils.create_folder(self.audio_directory)

        self.make_class_labels(config["midi_range"])

        self.max_real_time = config["max_real_time_mins"] * 60  # convert minutes to seconds

        files = glob.glob(self.dry_data_paths)
        print(f"Found {len(files)} files.")
        self.packed_paths = pack_paths(files)

        self.save_config()

    def save_config(self):
        shutil.copy(self.config_file, f"{self.out_dir}/generator_config_file.json")


class FeatureXtractConfig(Config):
    def parse_config(self, config_file):
        config = super().load_config(config_file)

        self.version = config["version"]
        self.chord_name = config["chord_name"]
        self.working_directory = config["working_directory"]

        self.sr = config["sr"]
        self.folds = config["folds"]
        self.is_mono = config["is_mono"]
        self.nfft = config["nfft"]
        self.nb_mel_bands = config["nb_mel_bands"]
        self.meta_filename = config["meta_filename"]
        self.fold_size = config["fold_size"]
        self.audible_threshold = config.get("audible_threshold", 2)

        self.make_directories(self.working_directory, self.version, self.chord_name)
        utils.create_folder(self.feature_folder)

        self.make_class_labels(config["midi_range"])

        self.win_len = self.nfft
        self.hop_len = self.win_len / 2

        self.save_config()

    def save_config(self):
        def parse_config(self, config_file):
            self.config_file = config_file

            print(f"Loading config from: {self.config_file}")

            with open(self.config_file) as json_file:
                config = json.load(json_file)

        shutil.copy(self.config_file, f"{self.out_dir}/fetature_config_file.json")


class TrainingConfig(Config):
    def parse_config(self, config_file):
        config = super().load_config(config_file)
        self.config = config

        self.experiment_name = config["experiment_name"]
        self.restart_checkpoint = config.get("restart_checkpoint_path", False)
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.folds = config["folds"]
        self.feature_folder = config["feature_folder"]

        # self.working_directory = config["working_directory"]
        # self.version = config["data_version"]
        # self.chord_name = config["chord_name"]
        # self.make_directories(self.working_directory, self.version, self.chord_name)

        self.chpt_folder = config["chpt_folder"]
        utils.create_folder(self.chpt_folder)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.make_class_labels(config["midi_range"])

        utils.create_folder(f"{self.chpt_folder}/{self.experiment_name}/")


class EvalConfig(Config):
    def parse_config(self, config_file):
        # self.working_directory = config["working_directory"]
        # self.version = config["data_version"]
        # self.chord_name = config["chord_name"]
        # self.make_directories(self.working_directory, self.version, self.chord_name)

        config = super().load_config(config_file)
        self.config = config

        self.experiment_name = config["experiment_name"]
        self.outputs = config['outputs']
        utils.create_folder(self.outputs)

        self.restart_checkpoint = config.get("restart_checkpoint_path", False)
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.folds = config["folds"]
        self.feature_folder = config["feature_folder"]

        self.chpt_folder = config.get('chpt_folder', False)
        self.checkpoint_range = config.get('checkpoint_range', False)

        self.checkpoints = []
        if self.chpt_folder and self.checkpoint_range:
            chpts = os.listdir(self.chpt_folder)
            for chpt in chpts:
                # number = int(list(filter(str.isdigit, chpts))[0])
                epoch = int(re.findall('[0123456789]+', chpt)[0])
                if min(self.checkpoint_range) < epoch < max(self.checkpoint_range):
                    self.checkpoints.append(f'{self.chpt_folder}{chpt}')
        else:
            self.checkpoints.append(self.restart_checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.make_class_labels(config["midi_range"])
