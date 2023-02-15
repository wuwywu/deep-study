import torch

# gpu环境配置
# torch.cuda.is_available()

# print(torch.__version__)
# print('gpu:', torch.cuda.is_available())

# 梯度下降算法(找最优解)
## 简单的线性回归 loss=(wx+b-y)**2

## 计算平均误差
def computer_error_for_line_give_point(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y-(w*x+b))**2
    
    return totalError/float(len(points))


## 用梯度算法迭代b, w
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # loss=(wx+b-y)**2 
        b_gradient = (2/N)*(w_current*x+b_current-y)
        w_gradient = (2/N)*(w_current*x+b_current-y)*x
        
    # 沿梯度下降
    new_b = b_current-(learningRate*b_gradient)
    new_w = w_current-(learningRate*w_gradient)
    return [new_b, new_w]

    