% This is the simplest implementation of a Neural Network to learn its fundamental concepts
% It is a feed-forward NN with a single hidden layer
% with 1-dimensional input x and 1-dimensional output y

% Note: here we use the Neural Network Matlab toolbox

% The output is 

%  y = sum_i w2_i * f(w1_i*x+b1_i) + b2;  with f the nonlinear activation function

% Purpose of exercise 2: 
% 1) Convince ourself that a NN is an universal approximator
% 2) Understand the role of the optimizer


% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

f_true =@(x) (x.^3-2*x.^2+1).*sin(8*pi*x); % complex analytical 'ground truth' function 
x = linspace(-1,1,100); 
t = f_true(x); 


num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 200; % max number of iterations

rng(1); % random seed for reproducibility

net1=feedforwardnet(num_neurons);
net2=feedforwardnet(num_neurons);

net1.trainFcn = 'trainbfg'; % quasi-newton method
net1.trainParam.epochs = 0;
net1.divideFcn='dividetrain';
net1.trainParam.showWindow = 0;

net2.trainFcn = 'trainlm'; % Levenberg-Marwardt backpropaagation
net2.trainParam.epochs = 0;
net2.divideFcn='dividetrain';
net2.trainParam.showWindow = 0;

fig=figure;
fig.Position=[230 200 1000 700];

net1=init(net1);
net2=init(net2);


net1=train(net1,x,t);net1.trainParam.epochs=1; % initialize net1
net2=train(net2,x,t);net2.trainParam.epochs=1; % initialize net2

time1=0;
time2=0;

for i=1:num_iter
    tic,net2=train(net2,x,t); time2=time2+toc;
    y2=net2(x);
    tic, net1=train(net1,x,t); time1=time1+toc;
    y1=net1(x);

    L1 = mean((y1 - t).^2); % MSE loss function (mean squared error)
    L2 = mean((y2 - t).^2); % MSE loss function (mean squared error)

% make a figure showing the NN fit and the decay of the loss function

    subplot(2,2,1)
    plot(x,t,'linewidth',2), hold on
    plot(x,y1,'ok','linewidth',2), hold off
    axis([-1 1 -1 1]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
    title(['L = ' num2str(L1) ' iter = ' num2str(i)])
    
    subplot(2,2,2)
    plot(x,t,'linewidth',2), hold on
    plot(x,y2,'or','linewidth',2), hold off
    axis([-1 1 -1 1]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
    title(['L = ' num2str(L2) ' iter = ' num2str(i)])
    
    subplot(2,2,3:4)
    semilogy(time1,L1,'ok'),hold on
    semilogy(time2,L2,'or')
    set(gca,'fontsize',16),xlabel('Compute time'),ylabel('Loss function')
    legend('bfg','lm')
    pause(0.01)

  
end
