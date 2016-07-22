% load MNIST dataset
training_data = loadMNISTImages('train-images-idx3-ubyte');
training_labels = loadMNISTLabels('train-labels-idx1-ubyte');
val_data = loadMNISTImages('t10k-images-idx3-ubyte');
val_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% Show an image.
% just_for_show = reshape(training_data, [28, 28, 60000]);
% imshow(just_for_show(:,:,1));

%train_num:训练数据行数，dimension：训练数据维度，val_num：测试数据个数
[train_num, dimension] = size(training_data);
val_num = size(val_data, 1);
%求均值，后续归零化
mean = sum(training_data)/train_num;
training_data = training_data - repmat(mean, [train_num, 1]);
val_data = val_data - repmat(mean, [val_num, 1]);

% 设置超参数，batch_size批尺寸即每次交互处理多少数据
hyperparams.batch_size = 100;
hyperparams.num_epochs = 5; % 迭代次数
hyperparams.reg = 0.0001; % L2范数
hyperparams.momentum = 0.9;%冲量单元
hyperparams.learning_rate = 0.1%学习步长;

%三层，128->128->10
filter_size = [128, 128, 10];

% initialize the model
%model指的应该是神经网络的系数初始化
model = init( dimension, filter_size );

% Train the model.
% You could achieve around 98% accuracy in 11 seconds (Intel i7-4790k).
%tic;
[model, training_loss, test_acc] = solver(training_data, training_labels, val_data, val_labels, model, hyperparams);
%toc;
% save the model.
save(['model.mat'],'model');

% plot
iterations_per_epoch = train_num / hyperparams.batch_size;
num_iters = hyperparams.num_epochs * iterations_per_epoch;

figure(1);hold on;
plot([1:num_iters], training_loss, 'Color', [0.14, 0.73, 0.06], 'LineStyle', '-', 'lineWidth', 1.5);
ylabel('traning loss');
xlabel('iteration')
hold off;

figure(2);hold on;
plot([1:hyperparams.num_epochs], test_acc, 'Color', [0.14, 0.73, 0.06], 'LineStyle', '-', 'lineWidth', 1.5);
ylabel('test accuracy');
xlabel('epoch');
hold off;

fprintf('done!\n');
