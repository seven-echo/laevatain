%% Laevatain is for Large-scale Heterogeneous Graph Representation Learning
% step 1: ���벻ͬ���͵��ڽӾ���
% step 2: ���ɲ�ͬԪ·������
% step 3: �������ͶӰѧϰԪ·����������Ծ���ĵ�ά��ʾ
% step 4: ����self-attention���������Ա�ʾΪ���ļ����ں�Ȩ��
% step 5: �õ����յı�ʾ

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
% Dataset = "C:\Users\liu2h1jun\Desktop\�칹����\CODE\���ݼ��������\DBLP\DBLP_full.mat";
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

[N,~] = size(FIRST{1,1}); % Ŀ�����ͽڵ����
q = 1; % ��ʼ������Ԫ·�������cell���
s = 3; % ͶӰ����ϡ����
d = 128; % dimension of embedding

% �ٶ��õ��ľ����Ϊ���ࣺ
% ��ʼ/����������Ŀ�����ͽڵ�Ϊ���������ڽӾ�������ΪFIRST_1,FIRST_2, ..., FIRST_i
% ����paper����֮������ù�ϵ(1��)�������û����û�֮���ת�������ޣ����۹�ϵ(3��)
% �ڶ���Ϊ����һ�����ӵ���Ŀ��������Ͳ�ͬ�Ķ���Ϊ���������ڽӾ�������ΪSECOND_1, ..., SECOND_j
% ����paper��author֮���д����ϵ(1��)�������û�����Ʒ֮��ĵ�����ղأ������ϵ(3��)
% ������Ϊ���Զ������ӵ�Ŀ�����͵Ľڵ�Ϊ���������ڽӾ�������ΪTHIRD_1, ..., THIRD_k
% ����paper��author�ٵ�organization(1��)�������̼ҵ���Ʒ�ٵ��û�֮��ı���������ղأ��������ϵ(3��)

NUM_1 = length(FIRST)+1;
NUM_2 = length(SECOND)+1;
NUM_3 = length(THIRD)+1;
FF = zeros(NUM_1, NUM_1);
FS = zeros(NUM_1, NUM_2);
ST = zeros(NUM_2, NUM_3);

%% step 2: �û�ά��FF,FS��ST������ʾ������һ�׾��󡢶��׾�������׾���֮�以��Ĺ�ϵ
% ������ݲ�ͬ�����ݼ����½���FF,FS,SC����������ã��������ֳ�����֮������ӹ�ϵ
% first to first һ�׹�ϵ֮�����ϵ
% �����û�����Ʒ֮���һ�׹�ϵ�����ж��֣���ʱ��Ҫ����FF����,ʹ�ò�ͬ���͵�һ�׹�ϵ֮��Ҳ�ܷ�������
FF(1,1) = 1; % for DBLP
FF(2,2) = 1;
FF(3,3) = 1;
FF(4,4) = 1;
% first to second ��һ�׹�ϵ�����ӵ����׹�ϵ�ľ���
% ����AP PV -> APVPA ��AP��FIRST(2), PV��SECOND(3) �� FS(2,3)=1;
% FS(1,1) = 1; % for DBLP

% second to third ͬFS����Ĺ���

i = sum(sum(FF~=0));
j = sum(sum(FS~=0));
k = sum(sum(ST~=0));
NUM = i+j+k;
meta = cell(NUM, 1); % ���ڴ洢Ԫ·������
meta_embedding = cell(NUM+1, 1); % ���ڴ洢Ԫ·��embedding
% ��ʱ�������ɵ�Ԫ·�������ǹ̶���(i+j+k)������i + FS��ST�з���Ԫ�ظ���
fprintf("Number of metapath: %d\n", NUM);
%% step 3: ����Ԫ·�����󲢽������ͶӰ
% output meta or meta_embedding
start_time = tic;

R = sparse_randmatrix(N,d,s); % ��Ϊ���е�Ԫ·����㶼��Ŀ��ڵ����ͣ������о����ΪN*N
% Ԫ·��������Ҫ�����Ի���Ҳ������Ҫ���Խ���Ԫ�����㣡���� ��Ҫ��
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
    ind_sc = find(FS(m,:)); % Ѱ�Ҷ�Ӧ��m��������Щ���׿ɴ�ڵ�����
    % PA*PA'*R
    for n = (1:j)
        temp_sec = R'*FIRST{m}*SECOND{ind_sc(n)};
        meta_embedding{q} = FIRST{m}* (SECOND{ind_sc(n)}*temp_sec'); % �ı���������ȼ����ӿ�����
        q = q + 1;
        
        k = sum(sum(ST(n,:)~=0));
        ind_th = find(ST(n,:)); % Ѱ�Ҷ�Ӧ��n��������Щ���׿ɴ�Ľڵ�����
        % PA*AV*AV'*PA'*R
        for o = (1:k)
            temp_thi = temp_sec*THIRD{ind_th(o)}; % ������Խ������״��ݣ�һ������������״��ݵĻ����Ͻ��еģ���ֱ��������
            meta_embedding{q} = FIRST{m}*(SECOND{ind_sc(n)}*(THIRD{ind_th(o)}*temp_thi'));
            q = q + 1;
        end
    end
end
clear FIRST SECOND THIRD temp_fri temp_thi temp_sec;
%% �������Ա�ʾ
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
%% step 4: ����attention �õ��ں�Ȩ��
% �������self attention�����ں�Ȩ�أ���Ҫ���ںϵľ���ֱ���NUM��meta-path��ʾ
% �����Ա�ʾ
% attention score���㷽����������ں�ϵ��
% �����Լ���
s_score = zeros(N,NUM+1);
% ѡ��һ����ʾ����Ϊ���ı�ʾchose_embedding������Ϊ����������attention score
% chose_embedding = att_embedding;
chose_embedding = meta_embedding{1,1};
for each = (1:NUM+1)
    if ~isempty(meta_embedding{each,1})
        s_score(:,each) = sum(chose_embedding .* meta_embedding{each,1},2)/d;
    end
end
% softmax������ÿ��s_score�õ���a_score�����Ϳ���ͨ��a_score���Ը�����ʾ�ϲ�
a_score = softmax(s_score')';

%% step 5: �ںϣ��õ����յı�ʾ
% ����a_score�ںϵõ���Ӧ�ı�ʾ
H = zeros(N,d);
for each = (1:NUM+1)
    if ~isempty(meta_embedding{each,1})
        H = H + meta_embedding{each,1}.*a_score(:,each);
    end
end
end_time = toc(start_time);
fprintf('time:%.2f s\n', end_time);
save(savepath, 'H');
