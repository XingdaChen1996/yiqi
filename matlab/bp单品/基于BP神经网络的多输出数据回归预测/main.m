%%  清空环境变量
%作者：程序小怪的小课堂
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%随机数种子固定结果
rng(2222)

%%  导入数据
res = readmatrix('回归数据.xlsx');

%%  数据归一化 索引
X = res(:,1:5);
Y = res(:,6:7);
x = mapminmax(X', 0, 1);
%保留归一化后相关参数
[y, psout] = mapminmax(Y', 0, 1);

%%  划分训练集和测试集
num = size(res,1);%总样本数
k = input('是否打乱样本(是：1，否：0)：');
if k == 0
   state = 1:num; %不打乱样本
else
   state = randperm(num); %打乱样本
end
ratio = 0.8; %训练集占比
train_num = floor(num*ratio);

x_train = x(:,state(1: train_num));
y_train = y(:,state(1: train_num));
trainnum = size(x_train, 2);

x_test = x(:,state(train_num+1: end));
y_test = y(:,state(train_num+1: end));
testnum = size(x_test, 2);

%%  创建网络
hiddens = 4; %隐藏层个数
% 激活函数（传递函数）：隐藏层为双曲正切函数，输出层为线性函数
tf = {'tansig', 'purelin'};
net = newff(x_train, y_train, hiddens,tf);

%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 
net.trainParam.goal = 1e-6;       % 误差阈值
net.trainParam.lr = 0.01;         % 学习率

%%  训练网络
net= train(net, x_train, y_train);

%%  仿真测试
re1 = sim(net, x_train);
re2 = sim(net, x_test );

%%  数据反归一化
%实际值
Y_train = Y(state(1: train_num),:)';
Y_test = Y(state(train_num+1:end),:)';

%预测值
pre1 = mapminmax('reverse', re1, psout);
pre2 = mapminmax('reverse', re2, psout);

%% 循环分别计算y
for i=1:size(Y,2)
    disp('        ')
    disp('        ')
    disp(['第',num2str(i),'个输出指标'])
    p1 = pre1(i,:);
    p2 = pre2(i,:);
    Y_tr = Y_train(i,:);
    Y_te = Y_test(i,:);
    
    %%  均方根误差
    error1 = sqrt(mean((p1 - Y_tr).^2));
    error2 = sqrt(mean((p2 - Y_te).^2));
    
    %% 相关指标计算
    % R2
    R1 = 1 - norm(Y_tr - p1)^2 / norm(Y_tr - mean(Y_tr))^2;
    R2 = 1 - norm(Y_te -  p2)^2 / norm(Y_te -  mean(Y_te ))^2;
    
    %  MAE
    mae1 = mean(abs(Y_tr - p1 ));
    mae2 = mean(abs(p2 - Y_te ));
    
    disp('训练集预测精度指标如下:')
    disp(['训练集数据的R2为：', num2str(R1)])
    disp(['训练集数据的MAE为：', num2str(mae1)])
    disp(['训练集数据的RMSE为：', num2str(error1)])
    disp('测试集预测精度指标如下:')
    disp(['测试集数据的R2为：', num2str(R2)])
    disp(['测试集数据的MAE为：', num2str(mae2)])
    disp(['测试集数据的RMSE为：', num2str(error2)])
    
    figure
    plot(1: trainnum, Y_tr, 'r-^', 1: trainnum, p1, 'b-+', 'LineWidth', 1)
    legend('真实值','预测值')
    xlabel('样本点')
    ylabel('预测值')
    title('训练集预测结果对比')
    
    %%画图
    figure
    plot(1: testnum, Y_te, 'r-^', 1: testnum, p2, 'b-+', 'LineWidth', 1)
    legend('真实值','预测值')
    xlabel('样本点')
    ylabel('预测值')
    title('测试集预测结果对比')
    
    %% 训练集百分比误差图
    figure
    plot((p1 - Y_tr )./Y_tr, 'b-o', 'LineWidth', 1)
    legend('百分比误差')
    xlabel('样本点')
    ylabel('误差')
    title('训练集百分比误差曲线')
    
    %% 测试集百分比误差图
    figure
    plot((p2 - Y_te )./Y_te, 'b-o', 'LineWidth', 1)
    legend('百分比误差')
    xlabel('样本点')
    ylabel('误差')
    title('测试集百分比误差曲线')
    
    %%  拟合图
    figure;
    plotregression(Y_tr, p1, '训练集', ...
                   Y_te, p2, '测试集');
    set(gcf,'Toolbar','figure');
end