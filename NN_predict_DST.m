% Simple implementation of a Neural Network applied to the prediction of the OMNI dataset


% Author: Enrico Camporeale (enrico.camporeale@colorado.edu)
% Date: 09/24/19

clear all, close all,
clc

omni=readtable('Omni_allyears.csv');

% define input features

X=[omni.Year omni.Decimal_Day omni.Hour omni.Bz_GSM omni.Proton_Density omni.Plasma_speed omni.DST_Index];

% remove Nans
[i,j]=find(isnan(X));
X(i,:)=[];

time=datetime(datevec(doy2date(X(:,2),X(:,1))))+hours(X(:,3)); % convert into date & time

% we want to make predictions for a given timelag

timelag = hours(1);
timediff = time(1+timelag/hours(1):end) - time(1:end-timelag/hours(1));
f=find(timediff==timelag);

t=X(f+timelag/hours(1),end); % we want to predict DST
X=X(f,4:end);
time=time(f);

% normalize inputs and outputx

X_mean = mean(X);
X_std = std(X);
X = (X-X_mean)./X_std;

t_mean = mean(t);
t_std = std(t);
t = (t-t_mean)./t_std;

% define the NN architecture
num_neurons=100; % define the number of neurons in the hidden layer
num_iter = 200; % max number of iterations

rng(1); % random seed for reproducibility

net=feedforwardnet(num_neurons);

inputset = size(X,1);

net.divideFcn='divideind';
net.divideParam.trainInd=[1:round(0.7*inputset)];
net.divideParam.valInd=[round(0.70*inputset)+1:round(0.85*inputset)];
net.divideParam.testInd=[round(0.85*inputset)+1:inputset];
net.trainParam.epochs = num_iter;
net.performParam.regularization = 0.1;

%net=init(net);
%net=train(net,X',t'); 

% to save time let us load a saved pre-trained network
load trained_DST_net.mat
y=net(X');

ytest=net(X(net.divideParam.testInd,:)');
ytrain=net(X(net.divideParam.trainInd,:)');
yval =net(X(net.divideParam.valInd,:)');

plot(time(net.divideParam.testInd),t(net.divideParam.testInd),'') 
hold on
plot(time(net.divideParam.testInd),ytest,'') 
legend('Ground truth','NN prediction')

figure
plotregression(t(net.divideParam.trainInd),ytrain,'Training set',...
t(net.divideParam.valInd),yval,'Validation set',...
t(net.divideParam.testInd),ytest,'Test set',...
t,y,'Whole set')


 


