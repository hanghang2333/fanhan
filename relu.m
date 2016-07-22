function [ y, cache ] = relu( x )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    cache = ( x < 0 );
    x( cache ) = 0;
    y = x;

end

