function [ loss, probs, grads ] = three_layer_net( X, model, label, reg )

    if(~exist('label'))
        label = [];
    end

    if(~exist('reg'))
        reg = 0.0001;
    end

    grads = {};%梯度
    loss = 0.0;%损失

    % Finetune the model
    w1 = model{1,1}; b1 = model{1,2};%从model参数里提取出参数
    w2 = model{2,1}; b2 = model{2,2};
    w3 = model{3,1}; b3 = model{3,2};

    % Forward pass%前向传播
    %fully_connected_layer的输出的a是那一层的输出，cache是那一层的输入时的参数
    [a1, cache1] = fully_connected_layer(X, w1, b1);
    %relu相当于将输入的小于0的部分全部替换为0，输出的cache是输入的哪些地方小于0的标志矩阵
    [a2, cache2] = relu(a1);

    [a3, cache3] = fully_connected_layer(a2, w2, b2);
    [a4, cache4] = relu(a3);

    [score, cache5] = fully_connected_layer(a4, w3, b3);
    %到第三层的输出了。对结果进行softmax损失函数计量
    [loss, probs, dscore] = your_loss_function(score,label,w1,w2,w3,reg);

    % Compute loss and gradients
    %这里regloss就是L2范数，进行的是对损失函数加上一个惩罚项以避免模型过于复杂。reg是前面定义的范数系数
    %这是原文件定义的正则项，我们不使用这个，改为自己写的。为了模块化，我们选择将w稀疏传入到your_loss_function函数里
    %使用那个模块统一处理，计算出损失函数值loss
    %reg_loss = 0.5 * (reg*sum(sum(w1.*w1)) + reg*sum(sum(w2.*w2)) + reg*sum(sum( w3.*w3 ))) ;

    if( reg == 0 )
        return;
    end
    %反向传播算法，之后对grad梯度进行计算，修正参数，grads里存储的是计算出来的梯度值
    % Backward pass
    [da4, dw3, db3] = fully_connected_layer_backward(dscore, cache5);
    da3 = relu_backward(da4, cache4);
    [da2, dw2, db2] = fully_connected_layer_backward(da3, cache3);
    da1 = relu_backward(da2, cache2);
    [~, dw1, db1] = fully_connected_layer_backward(da1, cache1);

    dw1 = dw1 + reg*w1;
    dw2 = dw2 + reg*w2;
    dw3 = dw3 + reg*w3;

    grads = {dw1, db1;
             dw2, db2;
             dw3, db3};

end
