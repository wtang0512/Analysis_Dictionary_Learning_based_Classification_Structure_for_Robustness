function [D,U,W,Q]=Distributed_sadl(Xk,Hk,Lk,anum,maxIter,lambda2,gamma1,gamma2,lambda1,xi1_bar,xi2_bar,xi3_bar,N_Cluster)
% min_{D,U,W,Q,epsilon1,epsilon2} \sum_t (1/2||D_t X_t-U_t||_2^2+theta||U_t||_1
%            +rho1_t/2||epsilon1_t||_2^2+rho2_t/2||epsilon1_t||_2^2
%            +delat1/2||Q_t||_2^2+delta2/2||W_t||_2^2
%            +lambda2/2||D_t||_2^2
%            +xi1_t/2||D-D_t||_2^2+xi2_t/2||Q-Q_t||_2^2+xi3_t/2||W-W_t||_2^2)
%       s.t. H_t=Q_t U_t+epsilon1_t
%            L_t=W_t(Q_t U_t)+epsilon2_t
%            ||d_t||_2^2=1
%            ||d||_2^2=1
%
% created by Wen 01/2020
% wtang6@ncsu.edu




%% Normalization for global dictionary D
Normalize = @(t) t./repmat(sqrt(sum(t.^2,2)),1,size(t,2)); 
%% Initialization
beta=2.5; %5.5
D = rand(anum,size(Xk{1},1));
U = zeros(anum,size(Xk{1},2));
Q = rand(size(Hk{1},1),size(U,1));
W = rand(size(Lk{1},1),size(Q,1));


Dk={};Qk={};Wk={};Uk={};
Y1k={};Y2k={};
tic;
for c=1:N_Cluster
    Dk{c,1}=D;%rand(anum,size(X,1));
    Uk{c,1}=U;%rand(anum,samplesize);
    Qk{c,1}=Q;%rand(size(H,1),size(U,1));
    Wk{c,1}=W;rand(size(Lk{1},1),size(Q,1));
    Y1k{c,1}=rand(size(Hk{c}));
    Y2k{c,1}=rand(size(Lk{c}));
    eta_Qk{c}=beta.*(norm(Qk{c},2).^2+norm(Wk{c},2).^2);
    eta_Wk{c}=beta.*(norm(Qk{c},2).^2);
    eta_Uk{c}=beta.*(norm(Qk{c},2).^2+norm(Wk{c}*Qk{c},2).^2);
    eta_Dk{c}=beta.*norm(Xk{c}'*Xk{c},2).^2;
end
fprintf("Initialization time = %f seconds. \n",toc);



% mu, rho and eta
mu = 1.25; % this one can be tuned
mu_bar = mu * 1e8;

xi1=1e-25;
xi2=1e-25;
xi3=1e-25;

rho = 1.03;%1.25;          % this one can be tuned

% other parameters
iter = 1;
converge = false;
%maxIter = 5000;
% norm_L = norm(L,'fro');

while ~converge
    iter = iter + 1;
    
    %for i=1:10
    parfor c=1:N_Cluster
        % update U
        Uk1{c} = -(Dk{c}*Xk{c}-Uk{c})./mu;
        Uk2{c} = -Qk{c}'*(Y1k{c}./mu+(Hk{c}-Qk{c}*Uk{c}));
        Uk3{c} = -(Wk{c}*Qk{c})'*(Y2k{c}./mu+(Lk{c}-Wk{c}*Qk{c}*Uk{c}));
        U_temp{c} = Uk{c} - (Uk1{c} + gamma1.*Uk2{c} + gamma2.*Uk3{c})./(eta_Uk{c});
        Uk{c} = ALst(U_temp{c}, lambda1./(mu*(eta_Uk{c})));
        
        %update Q
        Q_temp{c} = -gamma1.*(Y1k{c}./mu+(Hk{c}-Qk{c}*Uk{c}))*Uk{c}'-...
            gamma2.*Wk{c}'*(Y2k{c}./mu+(Lk{c}-Wk{c}*Qk{c}*Uk{c}))*Uk{c}'+...
            +xi2.*(Qk{c}-Q)./mu;
        Qk{c} = Qk{c} - Q_temp{c}./(eta_Qk{c});

        %update W
        W_temp{c} = -gamma2.*(Y2k{c}./mu + (Lk{c}-Wk{c}*Qk{c}*Uk{c}))...
           *(Qk{c}*Uk{c})'+xi3.*(Wk{c}-W)./mu;
        Wk{c} = Wk{c} - W_temp{c}./eta_Wk{c};
        
        Dk{c}=(Uk{c}*Xk{c}'+xi1.*D)*inv(Xk{c}*Xk{c}'+eye(size(Xk{c},1)).*(xi1+lambda2));
        
        Dk{c}=Dk{c}./repmat(sqrt(sum(Dk{c}.^2,2)),1,size(Dk{c},2));
         Y1k{c} = Y1k{c} + mu.*(Hk{c}-Qk{c}*Uk{c});
        Y2k{c} = Y2k{c} + mu.*(Lk{c}-Wk{c}*Qk{c}*Uk{c});
    end

    %update Global D
    D=mean( cat(3, Dk{:}), 3 );
    D=Normalize(D);
    
    %update Global Q
    Q=mean( cat(3, Qk{:}), 3 );
    
    %update Global W
    W=mean( cat(3, Wk{:}), 3 );
    
    
    % adaptive mu and rho
    mu = min(mu*rho, mu_bar);
    xi1=min(xi1*rho,xi1_bar);
    xi2=min(xi2*rho,xi2_bar);
    xi3=min(xi3*rho,xi3_bar);
    
%% Accuracy Tracing
%training accuracy
%     Lt=W*Q*(D*X); %cell2mat(Xk')
%     [~,label]=max(Lt);
%     Acc_Train(iter)=sum((Gtr-label)==0)./length(label);
%     disp(Acc_Train(iter));
%     
%testing accuracy   
%     Lt=W*Q*(D*Y);
%     
%     [~,label]=max(Lt);
%     Acc(iter)=sum((Gte-label)==0)./length(label);
%     disp(Acc(iter));
%     if max(acc,Acc(iter))~=acc
%         acc=max(acc,Acc(iter));
%         iteration=iter;
%     end
    
    if iter >= maxIter+1
        Tr = false;
        break
    end
    
end

end

