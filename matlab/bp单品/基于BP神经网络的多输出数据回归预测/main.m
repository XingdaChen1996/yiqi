%%  ��ջ�������
%���ߣ�����С�ֵ�С����
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%��������ӹ̶����
rng(2222)

%%  ��������
res = readmatrix('�ع�����.xlsx');

%%  ���ݹ�һ�� ����
X = res(:,1:5);
Y = res(:,6:7);
x = mapminmax(X', 0, 1);
%������һ������ز���
[y, psout] = mapminmax(Y', 0, 1);

%%  ����ѵ�����Ͳ��Լ�
num = size(res,1);%��������
k = input('�Ƿ��������(�ǣ�1����0)��');
if k == 0
   state = 1:num; %����������
else
   state = randperm(num); %��������
end
ratio = 0.8; %ѵ����ռ��
train_num = floor(num*ratio);

x_train = x(:,state(1: train_num));
y_train = y(:,state(1: train_num));
trainnum = size(x_train, 2);

x_test = x(:,state(train_num+1: end));
y_test = y(:,state(train_num+1: end));
testnum = size(x_test, 2);

%%  ��������
hiddens = 4; %���ز����
% ����������ݺ����������ز�Ϊ˫�����к����������Ϊ���Ժ���
tf = {'tansig', 'purelin'};
net = newff(x_train, y_train, hiddens,tf);

%%  ����ѵ������
net.trainParam.epochs = 1000;     % �������� 
net.trainParam.goal = 1e-6;       % �����ֵ
net.trainParam.lr = 0.01;         % ѧϰ��

%%  ѵ������
net= train(net, x_train, y_train);

%%  �������
re1 = sim(net, x_train);
re2 = sim(net, x_test );

%%  ���ݷ���һ��
%ʵ��ֵ
Y_train = Y(state(1: train_num),:)';
Y_test = Y(state(train_num+1:end),:)';

%Ԥ��ֵ
pre1 = mapminmax('reverse', re1, psout);
pre2 = mapminmax('reverse', re2, psout);

%% ѭ���ֱ����y
for i=1:size(Y,2)
    disp('        ')
    disp('        ')
    disp(['��',num2str(i),'�����ָ��'])
    p1 = pre1(i,:);
    p2 = pre2(i,:);
    Y_tr = Y_train(i,:);
    Y_te = Y_test(i,:);
    
    %%  ���������
    error1 = sqrt(mean((p1 - Y_tr).^2));
    error2 = sqrt(mean((p2 - Y_te).^2));
    
    %% ���ָ�����
    % R2
    R1 = 1 - norm(Y_tr - p1)^2 / norm(Y_tr - mean(Y_tr))^2;
    R2 = 1 - norm(Y_te -  p2)^2 / norm(Y_te -  mean(Y_te ))^2;
    
    %  MAE
    mae1 = mean(abs(Y_tr - p1 ));
    mae2 = mean(abs(p2 - Y_te ));
    
    disp('ѵ����Ԥ�⾫��ָ������:')
    disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
    disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
    disp(['ѵ�������ݵ�RMSEΪ��', num2str(error1)])
    disp('���Լ�Ԥ�⾫��ָ������:')
    disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])
    disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
    disp(['���Լ����ݵ�RMSEΪ��', num2str(error2)])
    
    figure
    plot(1: trainnum, Y_tr, 'r-^', 1: trainnum, p1, 'b-+', 'LineWidth', 1)
    legend('��ʵֵ','Ԥ��ֵ')
    xlabel('������')
    ylabel('Ԥ��ֵ')
    title('ѵ����Ԥ�����Ա�')
    
    %%��ͼ
    figure
    plot(1: testnum, Y_te, 'r-^', 1: testnum, p2, 'b-+', 'LineWidth', 1)
    legend('��ʵֵ','Ԥ��ֵ')
    xlabel('������')
    ylabel('Ԥ��ֵ')
    title('���Լ�Ԥ�����Ա�')
    
    %% ѵ�����ٷֱ����ͼ
    figure
    plot((p1 - Y_tr )./Y_tr, 'b-o', 'LineWidth', 1)
    legend('�ٷֱ����')
    xlabel('������')
    ylabel('���')
    title('ѵ�����ٷֱ��������')
    
    %% ���Լ��ٷֱ����ͼ
    figure
    plot((p2 - Y_te )./Y_te, 'b-o', 'LineWidth', 1)
    legend('�ٷֱ����')
    xlabel('������')
    ylabel('���')
    title('���Լ��ٷֱ��������')
    
    %%  ���ͼ
    figure;
    plotregression(Y_tr, p1, 'ѵ����', ...
                   Y_te, p2, '���Լ�');
    set(gcf,'Toolbar','figure');
end