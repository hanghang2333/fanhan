function [ model ] = init(dimension, filter_size )
%UNTITLED2 Summary of this function goes here
%normrnd是指定均值和标准差的正态分布，就是系数矩阵，依次为每一层的，
%b是一个简单的偏量，值为0.
%   Detailed explanation goes here

    w1 = normrnd(0, 0.01, [dimension, filter_size(1) ] );
    w2 = normrnd(0, 0.01, [filter_size(1), filter_size(2) ] );
    w3 = normrnd(0, 0.01, [filter_size(2), filter_size(3) ] );

    b1 = zeros(1, filter_size(1));
    b2 = zeros(1, filter_size(2));
    b3 = zeros(1, filter_size(3));

    model{1,1} = w1; model{1,2} = b1;
    model{2,1} = w2; model{2,2} = b2;
    model{3,1} = w3; model{3,2} = b3;

end
