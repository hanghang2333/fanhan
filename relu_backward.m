function [ dx ] = relu_backward( dy, cache )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    dy( cache ) = 0;
    dx = dy;

end

