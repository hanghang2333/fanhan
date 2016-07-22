function [ loss, probs, dscore ] = your_loss_function( score, label )
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
jihuo=0;
if jihuo=0
%使用的是sigmoid激活函数加log似然代价函数
temp_score=exp(-score)+1;
probs=1/temp_score;
sum_score=sum(sig_score,2);
sum_score=repmat(sum_score,1,size(score,2));
probs=probs./sum_score;}
elseif jihuo=1
%使用的是tanh激活函数
elseif jihuo=3
temp_score=exp(-2×score);
probs1=1-temp_score;
probs2=1+temp_score;
probs=probs1./probs2;
sum_score=sum(sig_score,2);
sum_score=repmat(sum_score,1,size(score,2));
probs=probs./sum_score;}
elseif jihuo=4

end
num = size(score, 1);
coord_x = [1:num]';
%使用的是log似然代价函数
k=log(probs);
k(label*size(probs,1) + coord_x )=0;
loss=-sum(k)/(10*num);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% implement the backward pass and compute the mean derivatives.
% store in dscore
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               your code
dscore = probs;
dscore( label*size(probs,1) + coord_x ) = dscore( label*size(probs,1) + coord_x )  - 1;
dscore = dscore/num;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
