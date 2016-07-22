function [ y, cache ] = fully_connected_layer( x, w, b )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    batch_size = size(x, 1);%这个batch_size指的是本次处理的数据的数目
%     x_temp = reshape(x, [num_of_images, length(x)/num_of_images] );
    y = x*w + repmat(b, [batch_size, 1]);%将偏量b扩展成batch_size*1的矩阵。y为这一层传播的输出
    cache.x = x;
    cache.w = w;
    cache.b = b;

end
