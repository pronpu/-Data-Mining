import os
import torch
from torch.utils.data import DataLoader
from models_loc import U_net
from net_util_loc import Dataset
from config_loc import config_loc as cfg


# 训练主函数入口
def train():
    print('Training started')

    save_epoch = cfg['save_epoch']
    min_avg_loss = float('inf')  # 将初始最小平均损失设置为正无穷

    # 模型
    model = U_net()

    # 读入权重
    start_epoch = 0

    # # 加载初始模型
    # if cfg['use_old_pkl']:
    #     model_path = os.path.join(cfg['path'], 'weights', cfg['old_pkl'])
    #     model.load_state_dict(torch.load(model_path))
    #     print(f'Model loaded from {model_path}')

    model.to(cfg['device'])
    model.summary()

    # 数据集
    dataset_path = os.path.join("D:\\heatmap", "data")
    dataset = Dataset(dataset_path)

    train_data_loader = DataLoader(dataset=dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True)

    # 优化器
    loss_F = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    max_loss_name = ''
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        total_loss = 0.0
        min_loss = float('inf')
        max_loss = 0.0

        # 按批次取文件
        for index, (img, label, img_name) in enumerate(train_data_loader):
            img = img.to(cfg['device'])
            label = label.to(cfg['device'])

            pre = model(img)  # 前向传播

            loss = loss_F(pre, label)  # 计算损失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()

            # 记录最小和最大损失
            if loss.item() < min_loss:
                min_loss = loss.item()

            if loss.item() > max_loss:
                max_loss = loss.item()
                max_loss_name = img_name

        avg_loss = total_loss / (index + 1)

        # 保存最小平均损失的模型
        if avg_loss < min_avg_loss:
            min_avg_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg['path'], "weights", 'min_loss.pth'))

        print(f'Epoch {epoch}, Images {index + 1}, Avg Loss {avg_loss:.6f}, Min Loss {min_loss:.6f}, Max Loss {max_loss:.6f}, Max Loss Image {max_loss_name}, Min Avg Loss {min_avg_loss:.6f}')
        print('-------------------')

        # 保存每个save_epoch的权重
        if (epoch + 1) % save_epoch == 0:
            save_name = f"epoch_{str(epoch + 1).zfill(3)}.pth"
            torch.save(model.state_dict(), os.path.join(cfg['path'], "weights", save_name))


if __name__ == "__main__":
    # 训练
    train()
