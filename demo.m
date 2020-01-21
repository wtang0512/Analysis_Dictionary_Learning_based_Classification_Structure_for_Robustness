%%demo of SADL Journal Version
% created in 09/2018 by Wen
% updated in 01/2020
% wtang6@ncsu.edu
%%
load('scene15.mat')
%% For SADL
X=training_feats; % training features
Y=testing_feats;  % testing features

%H is the structure of classes
%L is the label matrix
%Gtr is the label of training data
%Gte is the label of testing data

%hyper-parameters
lambda1=1e-3;
delta1=0.001;
delta2=0.001;
%maximum iteration
maxIter=283;
%number of atom, i.e., size of dictionary
anum=size(X,2);
% lambda4 regularization term of solving dictionary, tune it will affect the results.
lambda4=0.0003;
%gamma1,gamma2 will change the initialization of Y1 and Y2
gamma1=10;gamma2=4;
% a is the coefficient that guarantee the convergence,
% if the program go divergent, please make it larger.
a=5.5;


%training phase
fprintf('Training......\n');
tic;
[D,U,W,Q,T]=SADL(X,H,L,anum,maxIter,...
    lambda4,gamma1,gamma2,lambda1,delta1,delta2,a);
if ~isnan(T) && ~isinf(T)
    trainingtime=toc;
    fprintf('training time = %f\n',trainingtime);
    
    
    %testing phase
    fprintf('Testing......\n');
    tic;
    Lt=W*Q*(D*Y);
    [~,label]=max(Lt);
    Acc=sum((Gte-label)==0)./length(label);
    testingtime=toc;
    fprintf('testing time = %f\n',testingtime);
    fprintf('Classification Accuarcy = %f%% \n',Acc*100);
else
    
    fprintf('Divergence! Please increase the value of a to guarantee the convergence.\n');
    
end

%% For Distributed SADL
% set up the number of threads
N_Cluster=20;
maxNumCompThreads(N_Cluster);

%%  Hyper-parameters
% You need to tune them for different number of clusters.
% penalty of sparse coefficients
lambda1=1e-3;
% coefficient of L2 regularization of the dictionary
lambda2=0.5;
%controls of H=QU+e1 and L=W(QU)+e2
gamma1=6;gamma2=7;
%the penalty coefficients of three global communication terms
xi1_bar=1e-3;xi2_bar=1e-3;xi3_bar=1e-3;
%maximum iteration
maxIter=5;
%% propress
% subsample the datasets for each cluster to make sure that the samples of
% all different classes are chosen.
Xk={};Hk={};Lk={};
samplesize=size(X,2)/N_Cluster;

for i=1:N_Cluster
ck{i,1}=[i:N_Cluster:size(X,2)];
Hk{i,1}=H(:,ck{i,1});
Lk{i,1}=L(:,ck{i,1});
Xk{i,1}=X(:,ck{i,1});
end

%open the parallel paltform
parpool

fprintf('Distributed Training with %d clusters......\n',N_Cluster);
tic;
[D,U,W,Q]=Distributed_sadl(Xk,Hk,Lk,anum,maxIter,...
 lambda2,gamma1,gamma2,lambda1,xi1_bar,xi2_bar,xi3_bar,N_Cluster);
trainingtime=toc;
fprintf('training time = %f\n',trainingtime);

%The same testing phase
fprintf('Testing......\n');
tic;
Lt=W*Q*(D*Y);
[~,label]=max(Lt);
Acc=sum((Gte-label)==0)./length(label);
testingtime=toc;
fprintf('testing time = %f\n',testingtime);
fprintf('Classification Accuarcy = %f%% \n',Acc*100);