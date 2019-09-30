% This is the simplest implementation of a Neural Network to learn its fundamental concepts
% It is a feed-forward NN with a single hidden layer
% with 1-dimensional input x and 1-dimensional output y
% The output is 

%  y = sum_i w2_i * f(w1_i*x+b1_i) + b2;  with f the nonlinear activation function

% Purpose of exercise 3: 
% 1) Convince ourself that a NN is not good at extrapolation


% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

f_true =@(x) (x.^3-2*x.^2+1).*sin(8*pi*x); % complex analytical 'ground truth' function 

% let us choose training data only on the external half of the domain!
x = [linspace(-1,-0.5,50) linspace(0.5,1,50)]; 
t = f_true(x); 


num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 100; % max number of iterations

rng(1); % random seed for reproducibility

net=feedforwardnet(num_neurons);

net.trainFcn = 'trainlm'; % Levenberg-Marwardt backpropaagation
net.trainParam.epochs = 1;
net.divideFcn='dividetrain';
net.trainParam.showWindow = 0;

fig=figure;
fig.Position=[230 200 500 700];

net=init(net);

for i=1:num_iter
    net=train(net,x,t); 
    y=net(x);

    L = mean((y - t).^2); % MSE loss function (mean squared error)

    % make a figure showing the NN fit and the decay of the loss function

    subplot(2,1,1)
    plot(linspace(-1,1),f_true(linspace(-1,1)),'linewidth',2), hold on
    plot(linspace(-1,1),net(linspace(-1,1)),'','linewidth',2), hold off
    axis([-1 1 -8 8]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
    title(['L = ' num2str(L) ' iter = ' num2str(i)])
    
    subplot(2,1,2)
    semilogy(i,L,'ok'),hold on
    set(gca,'fontsize',16),xlabel('Iter'),ylabel('Loss function')
    pause(0.01)

  
end
