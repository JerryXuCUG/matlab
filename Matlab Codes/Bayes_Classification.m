%BAYES_CLASSIFICATION 对裂缝进行二分类
% 格式与PNN相同 且不进行归一化

[~, ~, train_raw] = xlsread("砂岩裂缝样本.xlsx"); 
[~, ~, para_raw] = xlsread("归一化参数.xlsx"); 
Model_save_path = "SVM_model_sand.mat";
target_index = ["AC", "CAL", "CN", "DEN"];
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
        temp = 2;
    end
    Label(i-1) = temp;
end
Label = Label';

%% 模型训练和测试
[x_train, y_train, x_test, y_test] = split_train_test(train_data, Label, 2, 0.7);
class_err=bayes([y_train, x_train], x_test, 2);
res = linspace(0,0,length(y_test)); % 分类结果
pro = linspace(0,0,length(y_test)); % 分类概率
for i=1:length(class_err)
    if class_err(i,1)>=class_err(i,2)
        res(i)=1;
        pro(i)=class_err(i,1);
    else
        res(i)=2;
        pro(i)=class_err(i,2);
    end
end
res = res';
pro = pro';
[TP,TN,FP,FN] = Confusion_matrix(y_test,res,1,2);
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
 
%% 内部函数
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

function err=bayes(TrainData,TestData,class)
%TrainData=[ 分类 x1 x2 x3 x4 ....]
[m,n]=size(TrainData);

for i=1:class
    groupNum(i)=0;
    group(i)=0;
    for j=1:m
        if TrainData(j,1)==i
            group(i)=group(i)+1;
        end
    end

    if i==1
        groupNum(i)=group(i);
    else
        groupNum(i)=groupNum(i-1)+group(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算总平均值
% for j=1:n-1
% TotalMean(j)=0;
% for i=1:m
% TotalMean(j)=TotalMean(j)+yangben(i,j+1);
% end
% TotalMean(j)=TotalMean(j)/m;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GroupMean=[];
for i=1:class
    if i==1
        low=1;
        up=groupNum(i);
    else
        low=groupNum(i-1)+1;
        up=groupNum(i);
    end
    matrix=TrainData(low:up,:);
    MatrixMean=mean(matrix); %各分类组平均值
    GroupMean=[GroupMean;MatrixMean];
    for u=low:up
        for v=2:n
            C(u,v-1)=TrainData(u,v)-MatrixMean(v);
        end
    end
end

V=C'*C/(m-class);
V_inv=inv(V); %对矩阵V求逆
%%
GroupMean=GroupMean(:,2:n);
Q1=GroupMean*V_inv;

%%
for i=1:class
    lnqi(i)=log(group(i)/m);
    mat=GroupMean(i,:);
    Q2(i)=lnqi(i)-0.5*mat*V_inv*mat';
end
%%
% 待判别样本TestData=[x1 x2 x3 x4 ....]
[u,~]=size(TestData);
result=[];
for i=1:u
    x=TestData(i,:);
    yy=Q1*x'+Q2';
    result=[result yy];
end

[rows,cols]=size(result);
for i=1:cols
    tmp=0;
    mlljj=result(:,i);
    for j=1:rows
        tmp=tmp+exp(result(j,i)-max(mlljj));
    end
    for j=1:rows
        err(j,i)=exp(result(j,i)-max(mlljj))/tmp;
    end
end
err=err'; %后验概率
end