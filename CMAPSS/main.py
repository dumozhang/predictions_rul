import os
import gradio as gr
import torch
import sys

sys.path.append('./model')
import torch.nn as nn
from xlstm_faster import KANxLSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# 1. 模型定义（必须与训练时完全一致）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = KANxLSTM(14, 2, 1, "ms", batch_first=True,
                              proj_factor_slstm=4 / 3, proj_factor_mlstm=2)
        self.fc1 = torch.nn.Linear(420, 32)
        self.dr = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = torch.tanh(x.reshape(x.shape[0], -1))
        x = self.dr(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# 2. 模型加载函数
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("model/kanxlstm_model.pth", map_location=device, weights_only=True)

    model = Net()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device


####################################################
def process_targets(data_length, early_rul=None):
    """
    Takes datalength and earlyrul as input and
    creates target rul.
    """
    if early_rul == None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(early_rul * np.ones(shape=(early_rul_duration,)), np.arange(early_rul - 1, -1, -1))


def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    num_batches = np.int64(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    """
    This function takes test data for an engine as first input. The next two inputs
    window_length and shift are same as other functins.

    Finally it takes num_test_windows as the last input. num_test_windows sets how many examplles we
    want from test data (from last). By default it extracts only the last example.

    The function return last examples and number of last examples (a scaler) as output.
    We need the second output later. If we are extracting more than 1 last examples, we have to
    average their prediction results. The second scaler halps us do just that.
    """
    max_num_test_batches = np.int64(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows


def process_test_targets(data_length, true_rul, early_rul=None):
    """
    因为训练集中的数据是放电到出现故障，也就是说在最后一个cycle那里剩余寿命就是0了，所以用process_targets中的方式生成RUL标签。
    但测试集中的发动机并没有放电到出现故障，给定的测试集RUL标签是经历过测试集记录的放电过程后的剩余寿命，也就是测试集最后一个cycle的实时剩余寿命。
    基于这个理解，如果需要测试集全局寿命RUL标签，须结合测试集RUL标签自行定义其生成函数。
    生成测试集的RUL标签，最后一个时间步的RUL为给定的真实值。

    参数:
        data_length (int): 发动机的周期数（测试集长度）
        true_rul (int): 测试集最后一个时间步的真实RUL
        early_rul (int, optional): 类似训练集的early_rul参数，用于提前终止

    返回:
        np.ndarray: RUL标签数组
    """
    if early_rul is None:
        # 无early_rul时，生成从 true_rul+data_length-1 递减到 true_rul 的数组
        return np.arange(true_rul + data_length - 1, true_rul - 1, -1)
    else:
        # 计算有效递减区间长度
        decrement_length = early_rul - true_rul
        if decrement_length <= 0:
            return np.arange(true_rul + data_length - 1, true_rul - 1, -1)

        # 计算前段填充长度
        early_rul_duration = data_length - decrement_length

        if early_rul_duration <= 0:
            # 数据长度不足以填充early_rul，直接生成递减部分
            return np.arange(early_rul - 1, early_rul - 1 - data_length, -1)
        else:
            # 前段填充early_rul，后段递减到true_rul
            part1 = np.full(shape=early_rul_duration, fill_value=early_rul)
            part2 = np.arange(early_rul - 1, true_rul - 1, -1)
            return np.concatenate([part1, part2])


#####################################################这个没用
# --- 数据预处理 ---
def preprocess_data(engine_id):
    # 加载测试数据
    test_data = pd.read_csv("data/test_FD001.txt", sep="\s+", header=None)
    engine_data = test_data[test_data[0] == engine_id].iloc[:, 2:].values

    # 标准化（使用训练时的scaler）
    scaler = StandardScaler()
    scaler.fit(pd.read_csv("data/train_FD001.txt", sep="\s+", header=None).iloc[:, 2:])
    scaled_data = scaler.transform(engine_data)

    # 生成窗口
    window_length = 30
    windows = [scaled_data[i:i + window_length] for i in range(len(scaled_data) - window_length + 1)]
    return np.array(windows)


#################################################
# 4. Gradio预测函数
def predict(engine_id):
    try:
        engine_id = int(engine_id)

        test = 1

        train_file = f'train_FD00{test}.txt'
        test_file = f'test_FD00{test}.txt'
        rul_file = f'RUL_FD00{test}.txt'

        train_dataset_path = os.path.join(r'data', train_file)
        train_data = pd.read_csv(train_dataset_path, delimiter='\\s+', header=None)

        # Load the test dataset as a dataframe
        test_dataset_path = os.path.join(r'data', test_file)
        test_data = pd.read_csv(test_dataset_path, delimiter='\\s+', header=None)

        rul_dataset_path = os.path.join('data', rul_file)
        true_rul = pd.read_csv(rul_dataset_path, delimiter='\\s+', header=None)

        window_length = 30
        shift = 1
        early_rul = 125
        processed_train_data = []
        processed_train_targets = []

        # How many test windows to take for each engine. If set to 1 (this is the default), only last window of test data for
        # each engine is taken. If set to a different number, that many windows from last are taken.
        # Final output is the average output of all windows.
        num_test_windows = 5
        processed_test_data = []
        num_test_windows_list = []

        columns_to_be_dropped = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

        train_data_first_column = train_data[0]
        test_data_first_column = test_data[0]

        # Scale data for all engines
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data.drop(columns=columns_to_be_dropped))
        test_data = scaler.transform(test_data.drop(columns=columns_to_be_dropped))

        train_data = pd.DataFrame(data=np.c_[train_data_first_column, train_data])
        test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data])

        num_train_machines = len(train_data[0].unique())
        num_test_machines = len(test_data[0].unique())

        # Process training and test data sepeartely as number of engines in training and test set may be different.
        # As we are doing scaling for full dataset, we are not bothered by different number of engines in training and test set.

        # Process trianing data
        for i in np.arange(1, num_train_machines + 1):
            temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values

            # Verify if data of given window length can be extracted from training data
            if (len(temp_train_data) < window_length):
                print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
                raise AssertionError("Window length is larger than number of data points for some engines. "
                                     "Try decreasing window length.")

            temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
            data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data,
                                                                                        temp_train_targets,
                                                                                        window_length=window_length,
                                                                                        shift=shift)

            processed_train_data.append(data_for_a_machine)
            processed_train_targets.append(targets_for_a_machine)

        processed_train_data = np.concatenate(processed_train_data)
        processed_train_targets = np.concatenate(processed_train_targets)

        ### 测试集全局RUL预测
        processed_test_targets = []

        # 处理测试集数据
        processed_test_targets = []
        for i in [engine_id]:  # 处理发动机engine_id
            # 提取测试数据
            temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values

            # 提取该发动机的真实RUL（假设每行对应一个发动机）
            true_rul_value = true_rul.iloc[i - 1, 0]  # 根据实际数据结构调整索引

            # 检查窗口长度
            if len(temp_test_data) < window_length:
                print(f"Test engine {i} doesn't have enough data for window_length of {window_length}")
                raise AssertionError("Window length too large.")

            # 生成RUL标签（确保传入标量值）
            temp_test_targets = process_test_targets(
                data_length=temp_test_data.shape[0],
                true_rul=true_rul_value,  # 关键修复点：传入整数
                early_rul=early_rul  # 确保 early_rul 也是标量
            )
            test_data_for_a_machine, test_targets_for_a_machine = process_input_data_with_targets(temp_test_data,
                                                                                                  temp_test_targets,
                                                                                                  window_length=window_length,
                                                                                                  shift=shift)

            processed_test_data.append(test_data_for_a_machine)
            processed_test_targets.append(test_targets_for_a_machine)

        processed_test_data = np.concatenate(processed_test_data)
        processed_test_targets = np.concatenate(processed_test_targets)

        # Shuffle training data
        index = np.random.permutation(len(processed_train_targets))
        processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

        import torch
        from torch.utils.data import Dataset, DataLoader
        processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(
            processed_train_data,
            processed_train_targets,
            test_size=0.2,
            random_state=83)

        model, device = load_model()
        all_rul_pred = []
        # 预测
        # model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for prediction
            data = torch.from_numpy(processed_test_data).float().to(
                device)  # Convert the data to the correct type and device
            outputs = model(data)  # Forward pass
            rul_pred = outputs.cpu().numpy().reshape(-1)
        preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
        mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                                     for ruls_for_each_engine, num_windows in
                                     zip(preds_for_each_engine, num_test_windows_list)]
        all_rul_pred.append(mean_pred_for_each_engine)

        # 可视化
        # fig = plt.figure(figsize=(10, 6))
        # plt.plot(range(len(preds)), [true_rul]*len(preds), 'b--', label='True RUL')
        # plt.plot(preds, 'r-', label='Predicted RUL')
        # plt.title(f"Engine {engine_id} Prediction")
        # plt.xlabel("Time Cycle")
        # plt.ylabel("Remaining Useful Life")
        # plt.legend()

        fig = plt.figure(figsize=(10, 6))
        plt.plot(processed_test_targets, label='True RUL', marker='o', linestyle='-', color='blue')
        plt.plot(rul_pred, label='Predicted RUL', marker='x', linestyle='--', color='red')
        plt.xlabel('Cycle')
        plt.ylabel('Remaining Useful Life (RUL)')
        plt.title(f"Engine {engine_id} Prediction")
        plt.legend()
        plt.grid(True)

        # 计算指标
        last_pred = rul_pred[-1]
        error = abs(last_pred - processed_test_targets[-1])

        return fig, f"Final Prediction: {last_pred:.1f} | Error: {error:.1f}"

    except Exception as e:
        return None, f"Error: {str(e)}"


# 5. 创建交互界面
interface = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(choices=list(map(str, range(1, 101))), label="Engine ID"),
    outputs=[
        gr.Plot(label="Prediction Result"),
        gr.Textbox(label="Metrics")
    ],
    title="Aircraft Engine RUL Prediction",
    description="Select engine ID (1-100) to predict remaining useful life"
)

# 6. 启动应用（Kaggle需启用互联网）
if __name__ == "__main__":
    interface.launch(share=False)