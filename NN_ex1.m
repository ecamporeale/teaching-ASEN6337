% This is the simplest implementation of a Neural Network to learn its fundamental concepts
% It is a feed-forward NN with a single hidden layer
% with 1-dimensional input x and 1-dimensional output y
% The output is 

%  y = sum_i w2_i * f(w1_i*x+b1_i) + b2;  with f the nonlinear activation function

% Purpose of exercise 1: 
% 1) understand the role of the learning rate in a simple steepest descent backpropagation algorithm
% 2) Play with different activation functions

% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

f_true = @(x) x.^2; % very simple analytical 'ground truth' function 
x = linspace(-1,1,20); 
t = f_true(x); 

% define the activation function
nonlin = @(x) ((x>0) + 0.01*(x<=0)).*x; % leaky RELU 
nonlin_der =@(x)  (x>0) + 0.01*(x<=0); % derivative of nonlin for backpropagation

% try other activation functions:
%nonlin = @(x) max(x,0); % RELU  
%nonlin_der =@(x)  (x>0); % derivative of RELU  

%nonlin = @(x) tanh(x); % sigmoid
%nonlin_der =@(x)  1-tanh(x).^2; % derivative of sigmoid 


num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 500; % max number of iterations

rng(1); % random seed for reproducibility

w1 = (2*rand(num_neurons,1)-1); % initialize randomly the weights
w2 = (2*rand(num_neurons,1)-1); % initialize randomly the weights
b1 = (2*rand(num_neurons,1)-1); % initialize randomly the bias
b2 = 0; % second bias is just a scalar

fig=figure;
fig.Position=[230 200 500 700];

for i=1:num_iter % this is the main iteration loop


    v = nonlin(w1*x+b1); % output of hidden layer; size of v is (num_neurons x size of training set)
    y = w2'*v + b2; % output of NN
    L = mean((y - t).^2); % MSE loss function (mean squared error)

    % make a figure showing the NN fit and the decay of the loss function

    subplot(2,1,1)
    plot(x,t,'linewidth',2), hold on
    plot(x,y,'o','linewidth',2), hold off
    axis([-1 1 -0.5 1]),set(gca,'fontsize',16),xlabel('x'),ylabel('y')
    title(['L = ' num2str(L) ' iter = ' num2str(i)])
    subplot(2,1,2)
    semilogy(i,L,'.k'),hold on
    set(gca,'fontsize',16),xlabel('# iter'),ylabel('Loss function')
    pause(0.01)
    
    % calculate derivatives of loss function with respect to weights and biases
    % using the chain rule

    dLdy =(y-t);  
    dLdw2 = (dLdy*v')';

    dLdw1 = dLdy.*w2.*nonlin_der(w1*x+b1).*x;
    dLdw1 = sum(dLdw1')';
    
    dLdb1 = dLdy.*w2.*nonlin_der(w1*x+b1);
    dLdb1 = sum(dLdb1')';
    
     
    learn_rate=0.002;

    % update weights and biases

    w1 = w1 - learn_rate*dLdw1;
    w2 = w2 - learn_rate*dLdw2; 
    b1 = b1 - learn_rate*dLdb1; 
    b2 = b2 - learn_rate*sum(dLdy);

   
end

