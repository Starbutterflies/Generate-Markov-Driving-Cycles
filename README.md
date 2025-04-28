# Generate-Markov-Driving-Cycles

这是研究生期间完成的工作，用于生成具有代表性的地区工况。  
这些代码在Lin等人的研究基础上，又迈出了一小步：  
**Estimating regional air quality vehicle emission inventories: constructing robust driving cycles**.

---

## 核心思路

![图片1008611](https://github.com/user-attachments/assets/b5f81745-7654-431d-a827-73f9f6eadd5a)


本项目分为以下几个模块：

---

### (1) `into_bins.py`

#### 主要函数

- **`process_df`**  
  用来检验新片段是否符合条件。  
  - **参数**：  
    `df`：原始的片段  
    `new_speed`：新片段的衔接速度  
  - **返回**：拼接后的新`df`，或`None`

- **`generate_character`**  
  用来计算片段的各种参数。  
  - **参数**：  
    `new_df`：需要计算参数的片段  
  - **返回**：一个包含各种参数的矩阵，包括：

    ```
    max_speed             # 最大速度
    avg_speed             # 平均速度
    avg_running_speed     # 平均运行速度
    positive_acceleration # 平均正加速度
    neg_dec               # 平均减速度
    idle_SPEED_ratio      # 怠速占比时间
    acceleration_ratio    # 加速占比时间
    dec_ratio             # 减速占比时间
    road_power            # 平均VSP
    rms_acceleration      # RPA
    ```
#### 主要类
- **`into_bins`**  
  **是主类，用来完成迭代循环的主要类。**
  - **输入的外部参数**:  
    
    `origin_data`:经过处理后的原始数据。
    
    `total_frequency_df`:原始数据的频率矩阵。在其他地方处理。
    
    `database_character`:原始数据的运动学特征。在其他地方处理。
  - **输出**:  
    地方工况。
  - **主要方法**  
  执行流程:  
![image](https://github.com/user-attachments/assets/046ba316-1762-4e86-afd1-e66baeda5d98)
    - **`into_bins`**  
       用来将原始的df按照提前划分好的类切成小块  
      - **返回**：切分后的小seg，储存在内存中。  
    
    - **`apply_character_cluster`**  
      将切割后的片段聚类。  
      average_speed => 平均速度  
      maximum_speed => 最大速度  
      minimum_speed => 最小速度  
      acceleration_rates => 加速度不为0所占时间  
      - **返回**：不同片段所对应的类别
    
    - **`generate_markov_array_1d`**  
      生成状态转移矩阵  
      - **返回**：  
      生成的马尔科夫矩阵。在循环迭代时，依据这个矩阵中的概率，加权抽取下一个类别，紧接着，就从状态中筛选合适的片段。  
    - **`generate_driving_cycle_1d`**  
      是迭代用的方法。**是最核心的迭代函数**  
      执行的核心逻辑：  
      1.选取开始片段。start_list包含了所有可能的初始片段，再从中抽取一个做为开始片段。  
      2.选取合适的类别。通过状态转移矩阵，加权抽取下一个类别  
      3.计算最佳损失。将所有合适的片段都添加到原始工况上，生成多个不同的备选工况，再计算所有备选工况的rank，通过这个rank来决定最佳片段。（事实上是一种贪心算法）  
      4.重复1-3，直至满足预设条件。（比如长度达到1800时，停止迭代）
      - **返回**：  
      一条具有代表性的地区工况。
---

### (2) `main.py`  
用来聚类的文件  
#### 主要类  
- **`cluster_data`**  
  用于切割原始数据的那次聚类。  
  - **输入的外部参数**:  
    `path`:原始数据路径所在的文件夹  
  - **输出**:  
    包含聚类信息的csv  
  - **主要方法**  
    - **`generate_K`**  
    使用手肘法确定最佳的类别数。  
      - **返回**：  
       一张手肘法的图片，在主文件夹里面。  
    - **`cluster_`**  
    聚类主函数，将类别加入独立的一列，保存为csv    
      - **返回**：  
       包含了列的新数据。data.csv系列  
    - **`show_fig`**  
    绘图函数  
      - **返回**：  
       一张聚类图，显示它到底把什么聚为一类。  
---

### (3) `utils.py`

工具函数文件。
