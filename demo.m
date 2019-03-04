%%demo of SADL Journal Version
% created by Wen 9/2018
% wtang6@ncsu.edu

%%
load('scene15.mat')
%%
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