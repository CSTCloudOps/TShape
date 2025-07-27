import numpy as np
import matplotlib.pyplot as plt

def main():
   # 读取时间序列数据
   file_path = '/home/cnic/ea/datasets/UTS/UCR/1sddb40/test.npy'
   data = np.load(file_path)
   label_file_path = '/home/cnic/ea/datasets/UTS/UCR/1sddb40/test_label.npy'
   label_data = np.load(label_file_path)

   # 检查数据的形状
   print(f"Data shape: {data.shape}")
   print(f"Data shape: {label_data.shape}")

   # 假设数据是一个二维数组，其中每一行是一个时间序列
   # 我们可以绘制前几个时间序列作为示例
#    num_series_to_plot = min(30000, data.shape[0])  # 绘制前5个时间序列，如果数据不足5个则绘制所有数据

   begin = 15000
   end = 17900

#    plt.figure(figsize=(10, 6))
   with open('/home/cnic/ea/datasets/UTS/UCR/1sddb40/case.txt', 'w') as f:
      for i in range(begin, end):
         f.write(np.array2string(data[i])+" ")
#        plt.subplot(num_series_to_plot, 1, i + 1)
   with open('/home/cnic/ea/datasets/UTS/UCR/1sddb40/case_label.txt', 'w') as f:
      for i in range(begin, end):
         f.write(np.array2string(label_data[i])+" ")
   
#    plt.tight_layout()
#    plt.show()

if __name__ == "__main__":
   main()