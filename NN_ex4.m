% This is the simplest implementation of a Neural Network to learn its fundamental concepts
% It is a feed-forward NN with a single hidden layer
% with 1-dimensional input x and 1-dimensional output y
% The output is 

%  y = sum_i w2_i * f(w1_i*x+b1_i) + b2;  with f the nonlinear activation function

% Purpose of exercise 4: 
% 1) Understand overfitting and how to avoid it


% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

f_true =@(x) x.^2; % simple analytical 'ground truth' function 
x = 2*rand(1,100)-1;

t = f_true(x); 


num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 200; % max number of iterations

rng(1); % random seed for reproducibility

net=feedforwardnet(num_neurons);

net.trainFcn = 'trainbfg'; % 

% experiment #1
net.trainParam.epochs = 100;
net.divideFcn='dividetrain';
net.trainParam.showWindow = 0;

net=init(net);
net=train(net,x,t); 
y=net(x);
L = mean((y - t).^2); % MSE loss function (mean squared error)

fig=figure;
fig.Position=[230 200 1000 700];
subplot(2,1,1)
plot(linspace(-1,1),f_true(linspace(-1,1)),'linewidth',2), hold on
plot(linspace(-1,1,1000),net(linspace(-1,1,1000)),'','linewidth',2), 
axis([-1 1 -1 1]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
legend('Ground truth','NN approximation')
title(['L = ' num2str(L) ])
disp('Why the NN approximation is so bad, yet the error is so small?')
pause
plot(x,t,'ok','linewidth',2)
legend('Ground truth','NN approximation','Training data')
subplot(2,1,2)
hist(abs(net.IW{1}),100)
set(gca,'fontsize',16),xlabel('Weights')

% experiment #2: try a regularization term    
pause
disp('Experiment #2 with regularization')

net=init(net);
net.performParam.regularization = 0.1;
net.trainParam.epochs = 10;
num_iter=50;

fig=figure;
fig.Position=[230 200 1000 700];

for i=1:num_iter
    net=train(net,x,t); 
    y=net(x);

    L = mean((y - t).^2); % MSE loss function (mean squared error)
    subplot(2,1,1)
    plot(linspace(-1,1),f_true(linspace(-1,1)),'linewidth',2), hold on
    plot(linspace(-1,1,1000),net(linspace(-1,1,1000)),'','linewidth',2),
    plot(x,y,'ok','linewidth',2), hold off
    axis([-1 1 -1 1]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
    title(['L = ' num2str(L) ])
    legend('Ground truth','NN approximation','Training data')
    pause(0.1)
    subplot(2,1,2)
    hist(abs(net.IW{1}),100)
    xlabel('Weights')
    set(gca,'fontsize',16),xlabel('Weights')
end

