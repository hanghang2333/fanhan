function [ best_model, training_loss, test_acc ] = solver( training_data, labels, val_data, val_labels, model, hyperparams )

    batch_size = hyperparams.batch_size;%100
    num_epochs = hyperparams.num_epochs;%5
    reg = hyperparams.reg;%0.0001
    momentum = hyperparams.momentum;%0.9
    learning_rate = hyperparams.learning_rate;%0.1

    num = size(training_data, 1);%num为训练数据数目
    iterations_per_epoch = num / batch_size;%每一轮迭代次数？=数目/100；
    num_iters = num_epochs * iterations_per_epoch;%5×？=迭代次数

    training_loss = zeros(1, num_iters);%训练损失初始化为全0.为1×num_iters矩阵。
    test_acc = [];%准确度——测试
    best_acc = 0.0;%准确度——最佳
    best_model = {};

    % train for num_iters, 3000 for example, iterations
    for i = 1: num_iters
        fprintf('iter %d: ', i);
        index = randperm(num, batch_size);%重新随机排列，从1-num返回batch_size即100个数。
        data = training_data( index, : );%取出这些数。
        batch_labels = labels( index );%取出这些数的标签。

        [loss, ~, grads] = three_layer_net( data, model, batch_labels, reg );
        training_loss(i) = loss;%第i次迭代的损失
        fprintf('training loss %f; learning rate: %f\n', loss, learning_rate);
        for p = 1: numel(model)%numel返回给定数组里的元素个数。这里应该是对每一个参数进行修正了。
            if i == 1; momentum_cache{p} = 0; end%i=1是初始，初始化一下
            momentum_cache{p} = momentum * momentum_cache{p} - learning_rate * grads{p};
            grad = momentum_cache{p};
            model{p} = model{p} + grad;
        end
        %后面没看，感觉不用看，下面就是测试方法了，也不用修改
        % test the model for each epoch
        if rem(i , iterations_per_epoch) == 0%rem是求余函数
            num_val = size(val_data, 1);%num_val为测试数据的数目
            val_results = [];
            for j = 1: num_val/100
                val_patches = val_data( (j-1)*100 + 1: (j-1)*100 + 100, :);
                val_patches_labels = val_labels( (j-1)*100 + 1: (j-1)*100 + 100 );
                % you can output the val loss if you like
                [val_loss, val_scores, ~] = three_layer_net(val_patches, model, val_patches_labels, 0);
                [~, patches_results] = max(val_scores, [], 2);
                val_results = [val_results; patches_results];
                % fprintf('val loss: %f\n', val_loss);
            end
            % compute the accuracy
            val_results = val_results - 1;
            acc = mean(val_results == val_labels);
            test_acc = [test_acc, acc];

            % store the best result
            if acc > best_acc
                best_acc = acc;
                best_model = model;
            end

            % reduce the learning rate every two epochs
            if rem((i/iterations_per_epoch), 2) == 0
                learning_rate = learning_rate * 0.1
            end

            fprintf('Epoch %d test: test acc %f\n', i/iterations_per_epoch, acc);
            fprintf('The best test acc is: %f\n', best_acc);
        end % test
    end % traing

end
