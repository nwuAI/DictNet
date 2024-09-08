import scipy.io
import numpy as np


def compute_loss(landmarks1, landmarks2):
    # 确保landmarks具有相同的形状
    assert landmarks1.shape == landmarks2.shape, "Landmarks must have the same shape"

    # 计算每个landmark之间的欧氏距离
    distances = np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=-1))

    # 返回平均距离作为总损失
    loss = np.mean(distances)
    return loss


# 加载.mat文件
mat = scipy.io.loadmat('/mnt/f/Data_Set/AFLW2000/image00002.mat')
# mat现在是一个字典，你可以通过键来访问数据
data = mat['pt3d_68']  # 替换'variable_name'为你的变量名
print(data.shape)
for key in mat.keys():
    print("key:", key)
