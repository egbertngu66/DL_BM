import matplotlib.pyplot as plt
import numpy as np

def draw_loss(train_loss, test_loss):
    '''
    Args:
        train_loss: 训练集loss, list
        test_loss: 验证集loss, list
    '''
    assert(len(train_loss) == len(test_loss))
    fig, ax = plt.subplots() # 创建图实例
    x = range(len(train_loss))
    ax.plot(x, train_loss, label='Training loss')
    ax.plot(x, test_loss, label='Test loss')
    ax.set_xlabel('epoch') 
    ax.set_ylabel('Loss')
    ax.legend() #自动检测要在图例中显示的元素，并且显示

    plt.show() 


def draw_actual_vs_predict(actuals, predicts, label):
    assert(actuals.shape == predicts.shape)
    # 设置x轴和y轴的范围
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # 设置坐标刻度
    x_ticks = np.arange(0, 101, 20)
    y_ticks = np.arange(0, 101, 20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # 设置坐标轴名称
    plt.xlabel("Actual values(MPa)")
    plt.ylabel("Predicted values(MPa)")

    # 绘制对角线
    plt.plot(range(101), range(101), color="black")
    # 绘制散点图
    plt.scatter(actuals, predicts, c="blue", s=20)
    plt.text(5, 80, label, fontdict={'size': '16', 'color': 'black'})


    plt.show() 