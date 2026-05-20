clear
%% 导入数据 初始化
%参数顺序：井号 参数 LABEL
[~, ~, train_raw] = xlsread("白云质泥岩裂缝样本2.xlsx"); 
[~, ~, para_raw] = xlsread("归一化参数2.xlsx"); 
Model_save_path = "SVM_model_Bmud2.mat";
target_index = ["AC", "CAL", "SP"];
train_index = "";
for i=1:size(train_raw, 2)
    train_index(i) = string(train_raw{1, i});
end

para_index = "";
for i=1:size(para_raw, 2)
    para_index(i) = string(para_raw{1, i});
end

%% 数据归一化
Data = [];
target_train_index = linspace(0,0,length(target_index));
for i=1:length(target_index)
    % 查找对应原始数据的下标
    target_train_index(i) = matches(train_index, target_index(i));  
end
target_para_index = zeros(2, length(target_index));
for i=1:length(target_index)
    % 查找对应原始数据的下标
    temp = target_index(i) +"_AVG";
    target_para_index(1,i) = matches(para_index, temp); 
    temp = target_index(i) +"_STD";
    target_para_index(2,i) = matches(para_index, temp); 
end

% 这一步按井号进行归一化，消除部分标准化影响
train_data = zeros(length(train_raw)-1, length(target_index));
for i=2:length(train_raw)
    core = train_raw{i, 1};
    for j =1:length(target_train_index)
        temp_data =  train_raw{i, target_train_index(j)};
        for k = 2:length(para_raw)
            if strcmp(para_raw{k, 1}, core)
                mean = para_raw{k, target_para_index(1, j)};
                xigema = para_raw{k, target_para_index(2, j)};
                train_data(i-1, j) = (temp_data - mean) / xigema;
                break;
            end
        end
    end
end
% 这一步提取LABEL并转置为列向量
Label = linspace(0,0,length(train_raw)-1);
label_index = matches(train_index, "LABEL");
for i=2:length(train_raw)
    temp = train_raw{i, label_index};
    if abs(temp-0)<1e-5 %非0
        temp = 0;
    end
    Label(i-1) = temp;
end
Label = Label';

%% 模型训练和测试
[x_train, y_train, x_test, y_test] = split_train_test(train_data, Label, 2, 0.7);

c = cvpartition(length(x_train),'KFold',5);
sigma = optimizableVariable('sigma',[1e-5,1e1],'Transform','log');
box = optimizableVariable('box',[1e-3,1e2],'Transform','log');

minfn = @(z)kfoldLoss(fitcsvm(x_train, y_train,'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',z.box,...
    'KernelScale',z.sigma)); 

results = bayesopt(minfn,[sigma,box],'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus');

z = bestPoint(results);
SVMmodel = fitcsvm(x_train, y_train,'KernelFunction','rbf', ...
    'KernelScale',z.sigma,'BoxConstraint',z.box);
SVMmodel = fitPosterior(SVMmodel, x_train, y_train);
save(Model_save_path, 'SVMmodel');

[label_predict, score] = predict(SVMmodel, x_test);
[TP,TN,FP,FN] = Confusion_matrix(y_test,label_predict,1,0);
acc = (TN + TP) / (TN + TP + FP + FN);
recall = TP / (TP + FN);
precision = TP / (TP + FP);
F1_score = 2 * precision * recall / (precision + recall);

fprintf("混淆矩阵：\n");
fprintf("--------------------------------------\n");
fprintf("|           | 预测标签为0  | 预测标签为1 |\n");
fprintf("|-----------|------------|------------|\n");
fprintf("| 真实标签为0 |%6d      |%6d      |\n", TN, FP);
fprintf("|-----------|------------|------------|\n");
fprintf("| 真实标签为1 |%6d      |%6d      |\n", FN, TP);
fprintf("--------------------------------------\n");
fprintf("正确率为%.3f\n", acc);
fprintf("召回率为%.3f\n", recall);
fprintf("精确率为%.3f\n", precision);
fprintf("F1分数为%.3f\n", F1_score);

%% 绘图
figure
plot(1: length(y_test), y_test, 'r-*', 1: length(y_test), label_predict, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'SVM验证集预测结果对比'; ['准确率=' num2str(acc*100) '%']};
title(string)
grid

%% 混淆矩阵
figure
cm = confusionchart(y_test, label_predict);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';


%% 函数
function index = matches(string_arr, string)
    index = 0;
    for i = 1:length(string_arr)
        if strcmp(string_arr(i),string)
            index = i;
            break;
        end
    end
    if index == 0
        fprintf("ERROR！未查找到标识：%s\n", string);
        return;
    end
end

function [x_train, y_train, x_test, y_test] = split_train_test(x, y, k, ratio)
%split_train_test 分割训练集和测试集
% x：参数矩阵
% y：标签
% k：类别个数
% ratio：训练集占比
% return：训练集矩阵、训练集标签、测试集矩阵、测试集标签

m = size(x, 1);
y_labels = unique(y);
d = [1:m]';

x_train = [];
y_train = [];

for i = 1:k
    comm_i = find( y == y_labels(i));
    if isempty(comm_i)
        continue;
    end
    size_comm_i = length(comm_i);
    rp = randperm(size_comm_i);
    rp_ratio = rp(1:floor(size_comm_i * ratio));
    ind = comm_i(rp_ratio);
    x_train = [x_train; x(ind, :)];
    y_train = [y_train; y(ind, :)];
    d = setdiff(d, ind);
end

x_test = x(d, :);
y_test = y(d, :);

end

function [TP,TN,FP,FN] = Confusion_matrix(y_test,y_predict,true_label,false_label)
%Confusion_matrix 计算二分类问题的混淆矩阵
% y_test：验证集标签
% y_predict：预测标签
% true_label：正例label
% alse_label：反例label
% return：混淆矩阵
TP = 0; % 预测为正，真实也为正
TN = 0; % 预测为反，真实也为反
for i=1:length(y_test)
    if y_test(i) == y_predict(i) && y_test(i) == true_label
        TP = TP + 1;
    elseif y_test(i) == y_predict(i) && y_test(i) == false_label
        TN = TN + 1;
    end
end
% FP 预测为正，真实为反
% FN 预测为反，真实为正
FP = length(find(y_test==false_label)) - TN;
FN = length(find(y_test==true_label)) - TP;
end