clear
%% 导入数据 初始化
%参数顺序：井号 参数
[file,path] = uigetfile('*.xlsx');
if isequal(file,0)
   disp('User selected Cancel');
   return;
else
   disp(['User selected ', fullfile(path,file)]);
end
file_path = fullfile(path,file);
[~, ~, data_raw] = xlsread(fullfile(file_path)); 
clear path file;
param_index = ["AC", "CAL", "SP"];
class_index = "岩性";
% 是否定义归一化参数，如果需要自己定义，则设置auto_set_std_param = false;
% 并按:[AC_AVG、AC_STD、CAL_AVG、CAL_STD、CN_AVG、CN_STD、DEN_AVG、DEN_STD]设置参数
% AC_AVG代表该井目的层段的AC平均值，不区分岩性；STD代表标准差
% 例如:std_param = [331.832, 13.388, 25.896, 1.575, 22.52, 6.678, 2.438, 0.208];
% 如不需要自己定义，则设置auto_set_std_param = true; 此时std_param将被忽略
auto_set_std_param = false;
std_param = [335.9975,15.68873,25.47676,1.819065,80.10648,8.057371];

% 获取参数对应的下标
data_index = "";
for i=1:size(data_raw, 2)
    data_index(i) = string(data_raw{1, i});
end
param_data_index = linspace(0,0,length(param_index));
for i=1:length(param_index)
    % 查找对应原始数据的下标
    param_data_index(i) = matches(data_index, param_index(i));  
end
class_data_index = matches(data_index, class_index);

% 读取模型
Model = ["SVM_model","net_pnn"];
class = ["_SLsand2.mat","_ZCsand2.mat","_Xsand2.mat","_Fsand2.mat","_mud4.mat","_Bmud2.mat","_baiyunyan4.mat","_NZbaiyunyan2.mat"];
svm_model_SLsand = load(Model(1) + class(1));
pnn_model_SLsand = load(Model(2) + class(1));
svm_model_ZCsand = load(Model(1) + class(2));
pnn_model_ZCsand = load(Model(2) + class(2));
svm_model_Xsand = load(Model(1) + class(3));
pnn_model_Xsand = load(Model(2) + class(3));
svm_model_Fsand = load(Model(1) + class(4));
pnn_model_Fsand = load(Model(2) + class(4));
svm_model_mud = load(Model(1) + class(5));
pnn_model_mud = load(Model(2) + class(5));
svm_model_Bmud = load(Model(1) + class(6));
pnn_model_Bmud = load(Model(2) + class(6));
svm_model_baiyunyan = load(Model(1) + class(7));
pnn_model_baiyunyan = load(Model(2) + class(7));
svm_model_NZbaiyunyan = load(Model(1) + class(8));
pnn_model_NZbaiyunyan = load(Model(2) + class(8));

%% 获取归一化参数
% 测井参数提取
data_reshape = zeros(length(data_raw)-1, length(param_index));
for i=2:length(data_raw)
    for j =1:length(param_data_index)
        temp_data =  data_raw{i, param_data_index(j)};
        data_reshape(i-1, j) = temp_data;
    end
end
local_std_param = linspace(0,0,length(param_data_index)* 2);
if auto_set_std_param
    index = 1;
    for i = 1:length(param_data_index)
        local_std_param(index) = mean(data_reshape(:,i));
        index = index + 1;
        local_std_param(index) = std(data_reshape(:,i));
        index = index + 1;
    end
else
    for i = 1:length(std_param)
        local_std_param(i) = std_param(i);
    end
end
clear index;
%% 对参数进行归一化，同时处理数据输出预测结果
data_std = zeros(size(data_reshape,1), size(data_reshape,2));
for i=1:size(data_reshape,1)
    for j=1:size(data_reshape,2)
        data_std(i,j) = (data_reshape(i,j) - local_std_param(j*2-1))/local_std_param(j*2);
    end
end
clear data_reshape;
% 输出预测结果，首先按岩性进行划分
predict_res = zeros(size(data_std,1),3);
% 数据格式为[归一化数据，索引]
temp_data_SLsand = [];
temp_data_ZCsand = [];
temp_data_Xsand = [];
temp_data_Fsand = [];
temp_data_mud = [];
temp_data_Bmud = [];
temp_data_baiyunyan = [];
temp_data_NZbaiyunyan = [];
for i = 1:size(data_std,1)
    temp_data = data_std(i,:);
    % 查找岩性
     temp_class = string(data_raw{i+1, class_data_index});
    if contains(temp_class,"砂砾岩") 
        temp_data_SLsand = [temp_data_SLsand; [temp_data,i]];
    elseif contains(temp_class,"中-粗砂岩")
        temp_data_ZCsand = [temp_data_ZCsand; [temp_data,i]];
    elseif contains(temp_class,"细砂岩")
        temp_data_Xsand = [temp_data_Xsand; [temp_data,i]];
    elseif contains(temp_class,"粉砂岩")
        temp_data_Fsand = [temp_data_Fsand; [temp_data,i]];    
    elseif contains(temp_class,"泥岩")
        temp_data_mud = [temp_data_mud; [temp_data,i]];
    elseif contains(temp_class,"白云质泥岩")
        temp_data_Bmud = [temp_data_Bmud; [temp_data,i]];
    elseif contains(temp_class,"白云岩")
        temp_data_baiyunyan = [temp_data_baiyunyan; [temp_data,i]];
    elseif contains(temp_class,"泥质白云岩")
        temp_data_NZbaiyunyan = [temp_data_NZbaiyunyan; [temp_data,i]];
    else
        fprintf("ERROR！未查找到标识：%s,位于文件的第%d行\n", temp_class, i+1);
        return;
    end
end

% 砂砾岩段
if size(temp_data_SLsand, 1) >= 1
    [predict_label,socre] = predict(svm_model_SLsand.SVMmodel, temp_data_SLsand(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_SLsand(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_SLsand.net_pnn, temp_data_SLsand(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_SLsand(i,length(param_index)+1);
        if predict_label(i) == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 中-粗砂岩段
if size(temp_data_ZCsand, 1) >= 1
    [predict_label,socre] = predict(svm_model_ZCsand.SVMmodel, temp_data_ZCsand(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_ZCsand(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_ZCsand.net_pnn, temp_data_ZCsand(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_ZCsand(i,length(param_index)+1);
        if predict_label(i) == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 细砂岩段
if size(temp_data_Xsand, 1) >= 1
    [predict_label,socre] = predict(svm_model_Xsand.SVMmodel, temp_data_Xsand(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_Xsand(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_Xsand.net_pnn, temp_data_Xsand(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_Xsand(i,length(param_index)+1);
        if predict_label(i) == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 粉砂岩段
if size(temp_data_Fsand, 1) >= 1
    [predict_label,socre] = predict(svm_model_Fsand.SVMmodel, temp_data_Fsand(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_Fsand(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_Fsand.net_pnn, temp_data_Fsand(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_Fsand(i,length(param_index)+1);
        if predict_label(i) == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 泥岩段
if size(temp_data_mud, 1) >= 1
    [predict_label,socre] = predict(svm_model_mud.SVMmodel, temp_data_mud(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_mud(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_mud.net_pnn, temp_data_mud(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_mud(i,length(param_index)+1);
        if predict_label == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 白云质泥岩段
if size(temp_data_Bmud, 1) >= 1
    [predict_label,socre] = predict(svm_model_Bmud.SVMmodel, temp_data_Bmud(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_Bmud(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_Bmud.net_pnn, temp_data_Bmud(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_Bmud(i,length(param_index)+1);
        if predict_label == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end

end

% 白云岩段
if size(temp_data_baiyunyan, 1) >= 1
    [predict_label,socre] = predict(svm_model_baiyunyan.SVMmodel, temp_data_baiyunyan(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_baiyunyan(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_baiyunyan.net_pnn, temp_data_baiyunyan(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_baiyunyan(i,length(param_index)+1);
        if predict_label == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end

% 泥质白云岩段
if size(temp_data_NZbaiyunyan, 1) >= 1
    [predict_label,socre] = predict(svm_model_NZbaiyunyan.SVMmodel, temp_data_NZbaiyunyan(:,1:length(param_index)));
    for i = 1:length(predict_label)
        index = temp_data_NZbaiyunyan(i,length(param_index)+1);
        predict_res(index,1) = predict_label(i);
        predict_res(index,2) = max(socre(i,:));
    end
    predict_label = sim(pnn_model_NZbaiyunyan.net_pnn, temp_data_NZbaiyunyan(:,1:length(param_index))');
    predict_label = vec2ind(predict_label);
    for i = 1:length(predict_label)
        index = temp_data_NZbaiyunyane(i,length(param_index)+1);
        if predict_label == 2
            predict_res(index,3) = 0;
        else
            predict_res(index,3) = 1;
        end
    end
end
clear temp_class temp_data_SLsand temp_data_mud temp_data_ZCsand;
clear temp_data_Xsand temp_data_Fsand temp_data_Bmud;
clear temp_data_baiyunyan temp_data_NZbaiyunyan;

% 封装数据并输出
data_raw{1,size(data_raw,2)+1} = 'SVM预测分类';
data_raw{1,size(data_raw,2)+1} = 'SVM分类概率';
data_raw{1,size(data_raw,2)+1} = 'PNN预测分类';

for i = 1:size(predict_res,1)
    data_raw{i+1,size(data_raw,2)-2} = predict_res(i,1);
    data_raw{i+1,size(data_raw,2)-1} = predict_res(i,2);
    data_raw{i+1,size(data_raw,2)} = predict_res(i,3);
end

output_path = strrep(file_path, '.xlsx', '_predict.xlsx');
xlswrite(output_path, data_raw);
fprintf("DONE！输出至%s\n",output_path);

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