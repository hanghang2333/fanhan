function [ loss, probs, dscore ] = your_loss_function( score, label,w1,w2,w3,reg)
% input:
% score: score for each class and each example
% label: ground truth for each example
%
% output:
% loss: loss for examples of a batch size
% probs: probility for each class.
% dscore: mean derivatives with respect to the input of loss function

% To do
% implement the forward pass and compute loss and probility(or other forms).
% store them in loss, probs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               your code
%%！！--如果要运行的话，有两种方法：
%%！！1.从https://github.com/hanghang2333/fanhan   下载完整代码导入matlab路径，在matlab里输入run即可运行。
%%！！2.需要将原项目文件里three_layer_net.m的！！------
%%！！--调用softmax函数的语句改为调用your_loss_function。！！----------
%%！！--并且由于我们将正则惩罚项加在这个文件里，所以：需要将！！-------------
%%！！--three_layer_net.m里的这一过程去除。！！---------------------
%%！！--有运行问题请email:ansjmsmw@icloud.com !!----------------------
%------------使用的是tanh激活函数，不能合理运行-------------
%temp_score=exp(-2*score);
%probs1=1-temp_score;
%probs2=1+temp_score;
%probs=probs1./probs2;

%-----------使用的是sigmoid激活函数，不能合理运行-------------
%temp_score=exp(-score)+1;
%probs=1./temp_score;

%-----------使用的是softmax激活函数------------------------
max_score = max(score, [], 2);
temp_score = bsxfun(@minus, score, max_score);
probs = exp(temp_score);

%--------------下面是归一化过程----------------------------
sum_score = sum(probs, 2);
sum_score = repmat(sum_score, 1, size(score, 2));
probs = probs./sum_score;

probs_cache=probs;

num=size(score,1);
coord_x = [1:num]';
%----------------------------原始的似然对数损失函数实现----------------------
%按照对数似然损失函数定义，对每条数据进行处理，分类正确的那个p不变求log，分类错误的那个p需要1-pi后取概率
%这样就使得：分类正确的p大时，分类错误的p小，损失函数值小，反之，则大（绝对值）。
%save=zeros(size(probs,1),size(probs,2));
%save(labels*size(probs,1)+coord_x)=probs(label*size(probs,1)+coord_x);
%probs=1-probs;
%probs(labels*size(probs,1)+coord_x)=save(label*size(probs,1)+coord_x);
%k=log(probs);
%loss=-sum((sum(k))')/(10*num);

%---------------------------改进的对数似然损失函数实现------------------------
%改进的话考虑我们进行的分类问题，对于神经网络最后一层的输出进行取值的话，还是以p值最大的那个为取值，这样只要最终结果正确就可以。
%即：如果最后一层对应正确的结点的socre也恰是最大的，则可以看做没有损失。那么这一行数据就可以不用再求损失了，省了一些计算量。
%为了不至于降低准确度，这里将条件加强为：正确的结点的socre对于的p需要大于0.5.
%这样在损失函数里log可以少计算很多值。不过由于matlab语言特性，对于数学运算减少的时间在和多了一些逻辑语句情况下总时间并没有太多差别。
%下面是修改的对数似然损失函数
flag=reshape(probs(label*size(probs,1)+coord_x),size(probs,1),1);
cache=(flag>0.5);
count_tmp=100-sum(cache);
count=1;
nzero=zeros(1,count_tmp);
for n=1:100
  if cache(n)==0
    nzero(count)=n;
    count=count+1;
  end
end
save=zeros(size(probs,1),size(probs,2));
save(label*num+coord_x)=probs(label*num+coord_x);
probs=1-probs;
probs(label*num+coord_x)=save(label*num+coord_x);
k=log(probs(nzero,:));
%k=log(probs);
loss=-sum((sum(k))')/(10*num);

%------------------------计算正则项并得出最终loss--------------------------
%下面是L1范数实现的正则项损失
%reg_loss=reg*sum(sum(abs(w1))) + reg*sum(sum(abs(w2))) + reg*sum(sum(abs(w3)))
%下面是L2范数实现的正则项损失
reg_loss = 0.5 * (reg*sqrt(sum(sum(w1.*w1))) + reg*sqrt(sum(sum(w2.*w2))) + reg*sqrt(sum(sum( w3.*w3 )))) ;
loss = loss+reg_loss;

%将probs还原为经过激活函数计算后的结果probs
probs=probs_cache;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% implement the backward pass and compute the mean derivatives.
% store in dscore
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               your code
dscore = probs_cache;
dscore( label*size(probs_cache,1) + coord_x ) = dscore( label*size(probs_cache,1) + coord_x )  - 1;
dscore = dscore/num;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
