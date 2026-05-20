%% 导入数据 初始化
%参数顺序：井号 参数 LABEL
[~, ~, train_raw] = xlsread("砂岩裂缝样本.xlsx"); 
[~, ~, para_raw] = xlsread("归一化参数.xlsx"); 
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