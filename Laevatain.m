%% Laevatain is for Large-scale Heterogeneous Graph Representation Learning
% step 1: 载入不同类型的邻接矩阵
% step 2: 生成不同元路径矩阵
% step 3: 利用随机投影学习元路径矩阵和属性矩阵的低维表示
% step 4: 利用self-attention机制以属性表示为核心计算融合权重
% step 5: 得到最终的表示

%% step 1: load data
% for alibaba - user embedding

net_path = "E:\_dataset\ECommAI_data\data\final_data\20190610.mat";
i_feature_path = "E:\_dataset\ECommAI_data\data\final_data\item_feature.mat";
u_feature_path = "E:\_dataset\ECommAI_data\data\final_data\user_feature.mat";
savepath = "embedding\user_embedding.mat";
load(net_path);

% for alibaba user embedding
% load(u_feature_path);
% attribute = user_feature;
% FIRST = cell(4,1);
% FIRST{1,1} = buy;
% FIRST{2,1} = cart;
% FIRST{3,1} = clk;
% FIRST{4,1} = collect;

% for alibaba item embedding
load(i_feature_path);
attribute = item_feature;
FIRST = cell(4,1);
FIRST{1,1} = buy';
FIRST{2,1} = cart';
FIRST{3,1} = clk';
FIRST{4,1} = collect';
clear buy cart clk collect;
% ----------
SECOND = {};
THIRD = {};

% for DBLP - paper embedding
% Dataset = "C:\Users\liu2h1jun\Desktop\异构网络\CODE\数据集处理代码\DBLP\DBLP_full.mat";
% savepath = "embedding\author_embedding.mat";
% load(Dataset);
% 
% attribute = title;
% FIRST = {Network};
% SECOND = {author};
% THIRD = {};

% for DBLP author embedding
% attribute = [];
% FIRST = {author'};
% SECOND = {Network'};
% THIRD = {};

[N,~] = size(FIRST{1,1}); % 目标类型节点个数
q = 1; % 初始化储存元路径矩阵的cell序号
s = 3; % 投影矩阵稀疏性
d = 128; % dimension of embedding

% 假定得到的矩阵分为三类：
% 起始/结束对象以目标类型节点为行向量的邻接矩阵，命名为FIRST_1,FIRST_2, ..., FIRST_i
% 比如paper互相之间的引用关系(1类)，比如用户和用户之间的转发，点赞，评论关系(3类)
% 第二类为可以一阶链接到与目标对象类型不同的对象为行向量的邻接矩阵，命名为SECOND_1, ..., SECOND_j
% 比如paper与author之间的写作关系(1类)，比如用户和商品之间的点击，收藏，购买关系(3类)
% 第三类为可以二阶链接到目标类型的节点为行向量的邻接矩阵，命名为THIRD_1, ..., THIRD_k
% 比如paper到author再到organization(1类)，比如商家到商品再到用户之间的被点击，被收藏，被购买关系(3类)

NUM_1 = length(FIRST)+1;
NUM_2 = length(SECOND)+1;
NUM_3 = length(THIRD)+1;
FF = zeros(NUM_1, NUM_1);
FS = zeros(NUM_1, NUM_2);
ST = zeros(NUM_2, NUM_3);

%% step 2: 用户维护FF,FS和ST表，来表示网络中一阶矩阵、二阶矩阵和三阶矩阵之间互相的关系
% 必须根据不同的数据集重新进行FF,FS,SC三个表的配置，才能体现出各边之间的链接关系
% first to first 一阶关系之间的联系
% 比如用户和商品之间的一阶关系可能有多种，此时需要调整FF矩阵,使得不同类型的一阶关系之间也能发生传递
FF(1,1) = 1; % for DBLP
FF(2,2) = 1;
FF(3,3) = 1;
FF(4,4) = 1;
% first to second 从一阶关系能链接到二阶关系的矩阵
% 例如AP PV -> APVPA 若AP在FIRST(2), PV在SECOND(3) 则 FS(2,3)=1;
% FS(1,1) = 1; % for DBLP

% second to third 同FS矩阵的构建

i = sum(sum(FF~=0));
j = sum(sum(FS~=0));
k = sum(sum(ST~=0));
NUM = i+j+k;
meta = cell(NUM, 1); % 用于存储元路径矩阵
meta_embedding = cell(NUM+1, 1); % 用于存储元路径embedding
% 此时可以生成的元路径条数是固定的(i+j+k)条，即i + FS和ST中非零元素个数
fprintf("Number of metapath: %d\n", NUM);
%% step 3: 计算元路径矩阵并进行随机投影
% output meta or meta_embedding
start_time = tic;

R = sparse_randmatrix(N,d,s); % 因为所有的元路径起点都是目标节点类型，故所有矩阵皆为N*N
% 元路径矩阵需要消除自环，也就是需要将对角线元素置零！！！ 需要吗？
for m = (1:i)
    l = sum(sum(FF(m,:)~=0));
    ind_ff = find(FF(m,:));
    % PP*PP'*R
    for i_ff = (1:l)
        temp_fri = R'*FIRST{ind_ff(i_ff)};
        meta_embedding{q} = FIRST{m}*temp_fri';
        q = q + 1;
    end
    j = sum(sum(FS(m,:)~=0));
    ind_sc = find(FS(m,:)); % 寻找对应的m行中有哪些二阶可达节点类型
    % PA*PA'*R
    for n = (1:j)
        temp_sec = R'*FIRST{m}*SECOND{ind_sc(n)};
        meta_embedding{q} = FIRST{m}* (SECOND{ind_sc(n)}*temp_sec'); % 改变运算符优先级，加快运算
        q = q + 1;
        
        k = sum(sum(ST(n,:)~=0));
        ind_th = find(ST(n,:)); % 寻找对应的n行中有哪些三阶可达的节点类型
        % PA*AV*AV'*PA'*R
        for o = (1:k)
            temp_thi = temp_sec*THIRD{ind_th(o)}; % 如果可以进行三阶传递，一定是在上面二阶传递的基础上进行的，故直接拿来用
            meta_embedding{q} = FIRST{m}*(SECOND{ind_sc(n)}*(THIRD{ind_th(o)}*temp_thi'));
            q = q + 1;
        end
    end
end
clear FIRST SECOND THIRD temp_fri temp_thi temp_sec;
%% 计算属性表示
% input: attribute matrix of target node type
% output: attribute embedding of target node type
if length(attribute)~=0
    temp = sum(attribute'.^2).^.5;
    temp(find(temp==0))=0.000001;
    A_T = bsxfun(@rdivide, attribute', temp); % Normalize
    ATR = A_T*R; % pre-projection
    % block times
    split_num = 10;
    split = ones(1,split_num)*(N-mod(N,split_num))/split_num;
    split(split_num) = split(split_num) + mod(N,split_num);
    block = mat2cell(A_T',split);
    att_embedding = cell(split_num,1);
    for iter = (1:split_num)
        att_embedding{iter,1} = block{iter}*ATR;
    end
    att_embedding = cell2mat(att_embedding);
    % att_embedding = A_T'*ATR;

    meta_embedding{NUM+1,1} = att_embedding;
else
    meta_embedding{NUM+1,1} = [];
end
%% step 4: 计算attention 得到融合权重
% 这里参照self attention计算融合权重，需要被融合的矩阵分别是NUM个meta-path表示
% 和属性表示
% attention score计算方法，计算出融合系数
% 相似性计算
s_score = zeros(N,NUM+1);
% 选出一个表示来作为中心表示chose_embedding，以其为中心来计算attention score
% chose_embedding = att_embedding;
chose_embedding = meta_embedding{1,1};
for each = (1:NUM+1)
    if ~isempty(meta_embedding{each,1})
        s_score(:,each) = sum(chose_embedding .* meta_embedding{each,1},2)/d;
    end
end
% softmax计算由每个s_score得到的a_score，最后就可以通过a_score来对各个表示合并
a_score = softmax(s_score')';

%% step 5: 融合，得到最终的表示
% 利用a_score融合得到对应的表示
H = zeros(N,d);
for each = (1:NUM+1)
    if ~isempty(meta_embedding{each,1})
        H = H + meta_embedding{each,1}.*a_score(:,each);
    end
end
end_time = toc(start_time);
fprintf('time:%.2f s\n', end_time);
save(savepath, 'H');
