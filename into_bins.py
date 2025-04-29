import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import generate_new_characters
import random
import json
from joblib import Parallel, delayed
import time
from utils import generate_safd, cal_total_loss


def process_df(df, new_speed, seg):
    if np.abs(df["SPEED"].iloc[0] - new_speed) <= 2.5 and len(df) <= 100:
        return pd.concat([seg, df["SPEED"]]).reset_index(drop=True)
    else:
        pass


def generate_character(new_df):
    """
    :param new_df: 拼接好的，只有速度的矩阵
    :return:
    """
    max_speed_matrix = new_df.max(axis=0)  # 最大速度矩阵
    avg_speed_matrix = new_df.mean(axis=0)  # 最大平均速度矩阵
    positive_SPEED_values = new_df[new_df > 0]  # 正速度矩阵
    avg_running_speed_matrix = positive_SPEED_values.mean()
    acceleration_ = new_df.diff()

    positive_acceleration_values = acceleration_[acceleration_ > 0]
    positive_acceleration_matrix = positive_acceleration_values.mean()
    neg_dec_values = acceleration_[acceleration_ < 0]
    neg_dec_matrix = neg_dec_values.mean()
    speed_zeros_count = (new_df == 0).sum()
    total_non_nan = new_df.count()

    idle_SPEED_ratio_matrix = speed_zeros_count / total_non_nan

    acc_pos_count = (acceleration_ > 0).sum()
    dec_neg_count = (acceleration_ < 0).sum()
    acceleration_ratio_matrix = acc_pos_count / total_non_nan
    dec_ratio_matrix = dec_neg_count / total_non_nan
    road_power_array = (86.3 * new_df + 0.0459 * (new_df ** 3)) / 1000
    road_power_array = road_power_array.add(317 * positive_acceleration_values * new_df / 1000,fill_value=0)
    road_power_matrix = road_power_array.mean()
    rms_acceleration_matrix = (acceleration_ ** 2).mean()
    character_df = pd.DataFrame({"max_speed_matrix": max_speed_matrix,
                                 "avg_speed_matrix": avg_speed_matrix,
                                 "avg_running_speed_matrix ": avg_running_speed_matrix,
                                 "positive_acceleration_matrix": positive_acceleration_matrix,
                                 "neg_dec_matrix": neg_dec_matrix,
                                 "idle_SPEED_ratio_matrix": idle_SPEED_ratio_matrix,
                                 "acceleration_ratio_matrix": acceleration_ratio_matrix,
                                 "dec_ratio_matrix ": dec_ratio_matrix, "road_power_matrix": road_power_matrix,
                                 "rms_acceleration_matrix": rms_acceleration_matrix})
    return character_df.fillna(0)


class into_bins(object):
    def __init__(self):
        self.markov_array = None
        self.labels = None
        self.origin_data = pd.read_csv(r'./intermediate_df/data.csv')
        self.origin_data["diff"] = self.origin_data["label"].diff()  # 读取数据
        self.origin_data.fillna(0, inplace=True)
        self.seg_list = []
        self.character_list = []
        self.ordered_label = []
        self.total_frequency_df = pd.read_csv(r"./Frequency/frequency.csv")
        with open("info.json", "r") as file:
            str_ = file.read()
            file.close()
        self.database_character = pd.DataFrame([json.loads(str_)])
        self.seg_list_markov = []
        self.cpu_count = joblib.cpu_count()

    def into_bins(self):
        """
        :return:切成小块的bin
        """
        slice_ = self.origin_data[self.origin_data["diff"] != 0].index  # 切割点位
        print("正在进行切片操作！")
        for i in tqdm(range(len(slice_) - 1)):
            if slice_[i + 1] - slice_[i] == 1:
                df = self.origin_data.loc[slice_[i]]
                self.seg_list.append(df)
            else:
                df = self.origin_data.loc[slice_[i]:slice_[i + 1] - 1]
                self.seg_list.append(df)

    def generate_characters(self, df):
        """
        :param df:
        :return:
        """
        length = len(df)
        if length == 1:
            average_speed = df["SPEED"]
            maximum_speed = df["SPEED"]
            minimum_speed = df["SPEED"]
            if df["acceleration"] > 0:
                acceleration_rates = 1
            elif df["acceleration"] <= 0:
                acceleration_rates = 0
        else:
            average_speed = df["SPEED"].mean()
            maximum_speed = df["SPEED"].max()
            minimum_speed = df["SPEED"].min()
            acceleration_rates = len(df["acceleration"][df["acceleration"] > 0]) / length
        return average_speed, maximum_speed, minimum_speed, acceleration_rates

    def generate_K(self, df):
        """
        :return: 这个还是只能看啊
        """
        SSE = []
        for k in range(1, 15):
            estimator = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)  # 构造聚类器
            estimator.fit(df)
            SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
            print("完成一次循环！")
        X = range(1, 15)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(X, SSE, 'o-')
        plt.show()
        series = pd.Series(SSE).diff(-1) / pd.Series(SSE).diff(-1).sum()
        print(series)

    def apply_characters_cluster(self):
        """
        :return:
        """
        print("正在进行聚类！")
        segment_list = self.seg_list
        for df in tqdm(segment_list):
            self.character_list.append(self.generate_characters(df))
        character_df = pd.DataFrame(self.character_list,
                                    columns=["average_speed", "maximum_speed", "minimum_speed", "acceleration_rates"])
        pd.DataFrame(character_df).to_csv("character.csv", index=False)
        transformer = StandardScaler()
        character_df = transformer.fit_transform(character_df)
        # self.generate_K(character_df)
        estimator = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300)
        estimator.fit(character_df)
        self.labels = estimator.labels_  # 进行分类

        # character_df["labels"] =

        # pd.DataFrame(self.labels).to_csv("labels.csv", index=False)

    def generate_markov_array_1d(self):
        """
        :return:生成的一维马尔科夫矩阵，按照平均速度划分状态
        """
        seg_list = self.seg_list
        sum_array = np.zeros(max(self.labels) + 1, dtype=float)
        len_array = np.zeros(max(self.labels) + 1, dtype=float)
        for label, df in tqdm(list(zip(self.labels, seg_list))):
            sum_array[label] += df["SPEED"].sum()
            len_array[label] += len(df)
        avg_speed = sum_array / len_array
        new_rank = np.argsort(np.argsort(avg_speed))
        trans_ = dict(zip([i for i in range(len(new_rank))], new_rank))
        for label in self.labels:
            self.ordered_label.append(new_rank[label])
        self.markov_array = np.zeros([6, 6], dtype=np.int64)
        for i in range(len(self.labels) - 1):
            self.markov_array[self.ordered_label[i], self.ordered_label[i + 1]] += 1
        self.markov_array = self.markov_array / np.sum(self.markov_array, axis=0)
        print(self.markov_array)

    def generate_markov_array_2d(self):
        """
        :return:生成的二维马尔科夫矩阵。按照平均速度划分状态。
        """
        seg_list = [df for dfs in self.seg_list for df in dfs]
        sum_array = np.zeros(max(self.labels) + 1, dtype=float)
        len_array = np.zeros(max(self.labels) + 1, dtype=float)
        for label, df in list(zip(self.labels, seg_list)):
            sum_array[label] += df["SPEED"].sum()
            len_array[label] += len(df)
        avg_speed = sum_array / len_array
        new_rank = np.argsort(np.argsort(avg_speed))
        trans_ = dict(zip([i for i in range(len(new_rank))], new_rank))
        for label in self.labels:
            self.ordered_label.append(new_rank[label])
        state_df = pd.DataFrame({"State": self.ordered_label})
        state_df['Prev_State'] = state_df['State'].shift(1)
        state_df['Prev_Prev_State'] = state_df['State'].shift(2)
        transition_matrix = state_df.groupby(['Prev_Prev_State', 'Prev_State', 'State']).size().unstack(fill_value=0)
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        print(self.labels)

    def generate_driving_cycle_1d(self):
        """
        :return: 生成的新工况
        """
        old_seg_list = self.seg_list  # 修正标签
        new_seg_list = [[] for i in range(6)]
        for (seg, label) in zip(old_seg_list, self.ordered_label):  # 用于修正df和series
            if len(seg.shape) != 1:
                seg_ = seg.copy()
                seg_.loc[:, "label"] = np.full(seg.shape[0], label)
                new_seg_list[label].append(seg_)
            else:
                seg_ = seg.to_frame().T
                seg_.loc[:, "label"] = label
                seg_.columns = ['SPEED', 'acceleration', 'label', 'diff']
                new_seg_list[label].append(seg_)

        start_list = [df for df in new_seg_list[0] if df["SPEED"].iloc[0] == 0 and df["SPEED"].iloc[-1] > 2 and (
                df["SPEED"][df["SPEED"] > 0].shape[0] / df.shape[0]) > 0.5 and len(df) < 100]

        seg = random.choice(start_list)["SPEED"]  # 选取开始片段，开始迭代。先选取一个初始的seg
        self.seg_list_markov.append(seg)
        next_stage = 0
        new_speed = seg.iloc[-1]
        while True:
            probability_list = [row[next_stage] for row in self.markov_array]
            next_stage = random.choices(np.arange(6), weights=probability_list)[0]
            print(f"当前的转移概率{probability_list}")
            print(f'下一个状态{next_stage}')
            t0 = time.time()
            results = Parallel(n_jobs=-1)(
                delayed(process_df)(df, new_speed, seg) for df in new_seg_list[next_stage]
            )
            new_list = [result for result in results if result is not None]
            t1 = time.time()
            print(f'查找可用的片段所用时间：{t1 - t0}')
            if len(new_list) != 0:
                SAFD_list = Parallel(n_jobs=-1)(delayed(generate_safd)(df) for df in new_list)
                SAFD_loss = [cal_total_loss(df, self.total_frequency_df) for df in SAFD_list]
                new_df = pd.concat(new_list, ignore_index=True, axis=1)
            else:
                print("*" * 100)
                print("当前状态", next_stage)
                print("没有对应的东西亚撒西")
                print("*" * 100)
                continue
            t2 = time.time()
            print(f"拼接df所用时间：{t2 - t1}")
            chunks = np.array_split(new_df, self.cpu_count, axis=1)

            character_dfs = Parallel(n_jobs=-1)(
                delayed(generate_character)(chunk) for chunk in chunks
            )
            character_df = pd.concat(character_dfs)
            character_df["SAFD_loss"] = SAFD_loss
            t3 = time.time()
            print(f"计算相关参数所用时间:{t3 - t2}")
            new_id = np.average(
                np.abs(character_df - self.database_character.values.repeat(character_df.shape[0], axis=0)).rank(),
                axis=1).argmin()
            new_seg = new_df.loc[:, new_id].dropna()
            self.seg_list_markov.append(new_seg)
            seg = pd.concat([seg, new_seg]).reset_index(drop=True)
            print(seg)
            new_speed = seg.iloc[-1]
            print(new_speed)
            print(f"当前工况长度：{len(seg)}")
            if len(seg) >= 1000:
                if seg.iloc[-1] <= 1:
                    break
        seg.to_csv(f"./driving cycle/{random.randint(1, 10 ** 6)}.csv")
        return seg

    def run(self):
        self.into_bins()
        self.apply_characters_cluster()
        self.generate_markov_array_1d()
        self.generate_driving_cycle_1d()


def joblib_run(class_):
    class_.run()


if __name__ == '__main__':
    # class_list = [into_bins() for _ in range(50)]
    # Parallel(n_jobs=16)(delayed(joblib_run)(class_) for class_ in class_list)

    while True:
        generator = into_bins()
        generator.run()
        del generator
