import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class cluster_data:
    def __init__(self,path):
        self.K = 6
        self.Km = None
        self.path = path
        try:
            self.data_list = [pd.read_csv(os.path.join(path,name)) for name in os.listdir(path)]  # 读取数据，然后进行拼接
        except:
            self.data_list = [pd.read_csv(os.path.join(path, name),encoding="GBK") for name in os.listdir(path)]
        self.df = pd.concat(self.data_list).reset_index(drop=True)
        self.df = pd.DataFrame({"SPEED":self.df["SPEED"]})
        self.df["acceleration"] = self.df["SPEED"].diff()
        self.df = self.df.fillna(0)
        self.df_ = pd.DataFrame({"acceleration":self.df["acceleration"]})
        transformer = StandardScaler()
        self.std_df = pd.DataFrame(transformer.fit_transform(self.df_),columns=["acceleration"])  # "SPEED"

    def generate_K(self):
        """
        :return: 这个还是只能忍看啊
        """
        SSE = []
        for k in range(1, 15):
            estimator = KMeans(n_clusters=k,init='k-means++',n_init=10, max_iter=300)  # 构造聚类器
            estimator.fit(self.std_df)
            SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
            print("完成一次循环！")
        X = range(1, 15)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(X, SSE, 'o-')
        plt.show()
        series = pd.Series(SSE).diff(-1)/pd.Series(SSE).diff(-1).sum()
        print(series)
    
    def cluster_(self,df):
        """
        :return:按照簇分类的数据
        """
        self.Km = KMeans(n_clusters=6, init='k-means++',n_init=50, max_iter=500)
        self.Km.fit(df)
        # plt.scatter(self.df["SPEED"], self.df["acceleration"],c=self.Km.labels_)
        # plt.show()
        self.df["label"] = self.Km.labels_
        self.df.to_csv(r"./intermediate_df/data2.csv", index=False)

    def show_fig(self):
        plt.scatter(np.arange(self.df_.shape[0]),self.df_["acceleration"],c=self.Km.labels_)
        plt.savefig(r"./myplot.png")


if __name__ == '__main__':
    cd = cluster_data(path=r"./origin_data")
    # cd.generate_K()
    cd.cluster_(cd.std_df)
    # cd.show_fig()
