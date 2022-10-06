import copy
import os
import random

import numpy as np
import pandas as pd


def scale(df):
    mins = pd.read_csv('data/train_mins.csv', header=None)
    maxes = pd.read_csv('data/train_maxes.csv', header=None)
    min = np.array(mins.min(axis=0))
    max = np.array(maxes.max(axis=0))

    # Normalize the data
    df = (np.matrix(df) - min) / (
            max - min)

    return pd.DataFrame(df)


class Data:
    def __init__(self, args):
        self.data_dir = os.path.join(os.getcwd(), args.data_dir)
        self.start_end_clip = args.start_end_clip
        self.last_n_ticks = args.last_n_ticks
        self.snapshots = self.samples = self.test_files = []
        self.snapshots_per_day = args.snapshots_per_day
        self.snapshot_size = args.snapshot_size
        self.total_snapshots = args.tot_snapshots

        if args.rescale:
            self.load_and_scale()

        # If we want to resample snapshots or we don't have any saved
        if args.resample or not os.listdir(os.path.join(self.data_dir, 'clean_train_data')):
            self.load_unscaled_train_files()
            indexes = random.sample(range(len(self.samples)), self.total_snapshots)
            self.save_snapshots(indexes)

        self.load_snapshots()

    def load_and_scale(self):

        if os.path.exists(self.data_dir + "/maxes.csv"):
            os.remove(self.data_dir + "/maxes.csv")
        if os.path.exists(self.data_dir + "/mins.csv"):
            os.remove(self.data_dir + "/mins.csv")

        for f in os.listdir(os.path.join(self.data_dir, 'train_data')):
            self.min_max_scaling_compute(f)

    def min_max_scaling_compute(self, filename):

        df = copy.deepcopy(
            pd.read_csv(os.path.join(os.path.join(self.data_dir, 'train_data'), filename), header=None).iloc[
            self.start_end_clip:-self.start_end_clip, :]).reset_index(drop=True)

        mins = pd.DataFrame().append(pd.Series(df.to_numpy().min(axis=0)), ignore_index=True)
        maxes = pd.DataFrame().append(pd.Series(df.to_numpy().max(axis=0)), ignore_index=True)

        mins.to_csv(self.data_dir + '/mins.csv', index=False, header=False, mode='a')
        maxes.to_csv(self.data_dir + '/maxes.csv', index=False, header=False, mode='a')

    def select_windows(self, df, k, n):
        # Compute drawdown and keep only the windows with the biggest ones

        min = df.mid_price[self.last_n_ticks - 1:].rolling(k).min()
        max = df.mid_price[self.last_n_ticks - 1:].rolling(k).max()
        drawdown = max - min

        indx = drawdown.drop_duplicates().nlargest(n)

        for i in indx.index:
            self.samples.append((scale(df.iloc[i - k + 1 - self.last_n_ticks - 1:i + 1].drop(columns=['mid_price'])),
                                 df.iloc[i - k + 1 - self.last_n_ticks - 1:i + 1].drop(columns=['mid_price'])))

    def load_unscaled_train_files(self):
        for filename in os.listdir(os.path.join(self.data_dir, 'train_data')):
            df = copy.deepcopy(
                pd.read_csv(os.path.join(os.path.join(self.data_dir, 'train_data'), filename), header=None).iloc[
                self.start_end_clip:-self.start_end_clip, :].reset_index(drop=True))
            df['mid_price'] = abs(df.iloc[:, 0] + df.iloc[:, 2]) / 2
            self.select_windows(df, self.snapshot_size, self.snapshots_per_day)

    def load_test_file(self, i):
        filename = sorted(os.listdir(os.path.join(self.data_dir, 'test_data')))[i]
        df = pd.read_csv(os.path.join(os.path.join(self.data_dir, 'test_data'), filename), header=None).iloc[
             self.start_end_clip:-self.start_end_clip, :].reset_index(drop=True)
        return filename, [scale(df), df]

    def save_snapshots(self, indexes):
        for j, i in enumerate(indexes):
            self.samples[i][0].to_csv(os.path.join(self.data_dir, f'clean_train_data/scaled_{j}.csv'), index=False)
            self.samples[i][1].to_csv(os.path.join(self.data_dir, f'clean_train_data/unscaled_{j}.csv'), index=False)

        del self.samples

    def load_snapshots(self):
        for i in range(self.total_snapshots):
            scaled = pd.read_csv(os.path.join(self.data_dir, f'clean_train_data/scaled_{i}.csv'))
            unscaled = pd.read_csv(os.path.join(self.data_dir, f'clean_train_data/unscaled_{i}.csv'))
            self.snapshots.append([scaled, unscaled])
