# Generate-Markov-Driving-Cycles

这是研究生期间完成的工作，用于生成具有代表性的地区工况。  
这些代码还原了Lin等人的研究工作：  
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
- **`generate_character`**  
---

### (2) `main.py`

主程序文件。

---

### (3) `utils.py`

工具函数文件。
