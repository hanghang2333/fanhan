function [ dx, dw, db ] = fully_connected_layer_backward( dy, cache )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    x = cache.x;
    w = cache.w;
    b = cache.b;

    dx = dy*w';
    dx = reshape(dx, size(x));

%     num = size(x, 1);
%     x_temp = reshape(x, [num, length(x)/num] );
    dw = x'*dy;
    %每列分别求和
    db = sum(dy, 1);

end
