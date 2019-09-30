% This is the simplest implementation of a Neural Network to learn its fundamental concepts
% It is a feed-forward NN with a single hidden layer
% with 1-dimensional input x and 1-dimensional output y
% The output is 

%  y = sum_i w2_i * f(w1_i*x+b1_i) + b2;  with f the nonlinear activation function

% Purpose of exercise 4: 
% 1) Understand train, validation and test set


% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

f_true =@(x) (x.^3-2*x.^2+1).*sin(4*pi*x); % analytical 'ground truth' function 
x = linspace(-1,1);
t = f_true(x); 
t = t + randn(size(t))*0.25; % add some noise

num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 200; % max number of iterations

rng(1); % random seed for reproducibility

net=feedforwardnet(num_neurons);

net.trainFcn = 'trainbfg'; % 
net.divideFcn='divideind';
net.divideParam.trainInd=[1:75];
net.divideParam.valInd=[];
net.trainParam.epochs = 300;
net.performParam.regularization = 0.1;
net=init(net);

net=train(net,x,t); 
y=net(x);
L_train = mean((y(1:75) - f_true(x(1:75))).^2); % MSE loss function (mean squared error)
L_val = mean((y(76:100) - f_true(x(76:100))).^2); % MSE loss function (mean squared error)

fig=figure;
fig.Position=[230 200 1000 700];
subplot(2,1,1)
plot(x(1:75),t(1:75),'ok','linewidth',2), hold on
plot(x(76:end),t(76:end),'or','linewidth',2), hold on
plot(linspace(-1,1),f_true(linspace(-1,1)),'linewidth',2)
plot(linspace(-1,1,1000),net(linspace(-1,1,1000)),'','linewidth',2), 
axis([-1 1 -2 3]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
title(['All points in training set; L train = ' num2str(L_train) ' L val = ' num2str(L_val) ])
legend('Training data', 'Val Data','Ground truth','NN approximation','Location','NorthWest')


net.trainFcn = 'trainbfg'; % 
net.trainParam.epochs = 200;
net=init(net);
net=train(net,x(1:75),t(1:75)); 
net.trainParam.epochs = 100;
net.divideFcn='divideind';
net.divideParam.trainInd=[1:75];
net.divideParam.valInd=[76:100];
net.performParam.regularization = 0.1;
net.trainParam.showWindow = 1;
net.trainParam.max_fail=10;
net=train(net,x,t); 
y=net(x);

L_train = mean((y(1:75) - f_true(x(1:75))).^2); % MSE loss function (mean squared error)
L_val = mean((y(76:100) - f_true(x(76:100))).^2); % MSE loss function (mean squared error)

subplot(2,1,2)
plot(x(1:75),t(1:75),'ok','linewidth',2), hold on
plot(x(76:end),t(76:end),'or','linewidth',2), hold on
plot(linspace(-1,1),f_true(linspace(-1,1)),'linewidth',2)
plot(linspace(-1,1,1000),net(linspace(-1,1,1000)),'','linewidth',2), 
axis([-1 1 -2 3]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
title([' 75/25 points in train/val set;L train = ' num2str(L_train) ' L val = ' num2str(L_val) ])
legend('Training data', 'Val Data','Ground truth','NN approximation','Location','NorthWest')


