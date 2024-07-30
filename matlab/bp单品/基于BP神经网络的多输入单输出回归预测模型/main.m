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
X = res(:,1:end-1);
Y = res(:,end);
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
y_train = y(state(1: train_num));
trainnum = size(x_train, 2);

x_test = x(:,state(train_num+1: end));
y_test = y(state(train_num+1: end));
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
Y_train = Y(state(1: train_num))';
Y_test = Y(state(train_num+1:end))';

%Ԥ��ֵ
pre1 = mapminmax('reverse', re1, psout);
pre2 = mapminmax('reverse', re2, psout);

%%  ���������
error1 = sqrt(sum((pre1 - Y_train).^2) ./ trainnum);
error2 = sqrt(sum((pre2 - Y_test).^2) ./ testnum);

%% ���ָ�����
% R2
R1 = 1 - norm(Y_train - pre1)^2 / norm(Y_train - mean(Y_train))^2;
R2 = 1 - norm(Y_test -  pre2)^2 / norm(Y_test -  mean(Y_test ))^2;

%  MAE
mae1 = mean(abs(Y_train - pre1 ));
mae2 = sum(abs(pre2 - Y_test )) ./ testnum ;

disp('ѵ����Ԥ�⾫��ָ������:')
disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['ѵ�������ݵ�RMSEΪ��', num2str(error1)])
disp('���Լ�Ԥ�⾫��ָ������:')
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
disp(['���Լ����ݵ�RMSEΪ��', num2str(error2)])

figure
plot(1: trainnum, Y_train, 'r-^', 1: trainnum, pre1, 'b-+', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('������')
ylabel('Ԥ��ֵ')
title('ѵ����Ԥ�����Ա�')

%%��ͼ
figure
plot(1: testnum, Y_test, 'r-^', 1: testnum, pre2, 'b-+', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('������')
ylabel('Ԥ��ֵ')
title('���Լ�Ԥ�����Ա�')

%% ѵ�����ٷֱ����ͼ
figure
plot((pre1 - Y_train )./Y_train, 'b-o', 'LineWidth', 1)
legend('�ٷֱ����')
xlabel('������')
ylabel('���')
title('ѵ�����ٷֱ��������')

%% ���Լ��ٷֱ����ͼ
figure
plot((pre2 - Y_test )./Y_test, 'b-o', 'LineWidth', 1)
legend('�ٷֱ����')
xlabel('������')
ylabel('���')
title('���Լ��ٷֱ��������')

%%  ���ͼ
figure;
plotregression(Y_train, pre1, 'ѵ����', ...
               Y_test, pre2, '���Լ�');
set(gcf,'Toolbar','figure');
