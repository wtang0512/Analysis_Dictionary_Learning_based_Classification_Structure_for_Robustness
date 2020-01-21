% ||DX-U||_2^2+gamma1||H-QU||_^2+gamma2||L-W(QU)||_2^2+theta||U||_1
function [D,U,W,Q,T]=SADL(X,H,L,anum,maxIter,lambda3,gamma1,gamma2,theta,delta1,delta2,a)
% min_{D,U,W,Q,epsilon1,epsilon2} 1/2||DX-U||_2^2+theta||U||_1
%            +rho1/2||epsilon1||_2^2+rho2/2||epsilon1||_2^2
%            +delat1/2||Q||_2^2+delta2/2||W||_2^2
%            +lambda4/2||D||_2^2
%       s.t. H=QU+epsilon1
%            L=W(QU)+epsilon2
%
% created by Wen 09/2018
% wtang6@ncsu.edu

tol=1e-3;
Tr = true; % when Tr == false, there is no feasible solution

%a=2.5;

%% Initialization
D = rand(anum,size(X,1));
U = zeros(anum,size(X,2));
Q = rand(size(H,1),size(U,1));
W = rand(size(L,1),size(Q,1));

Y1 = rand(size(H));
Y2 = zeros(size(L));

% mu, rho and eta

mu=1.5/norm(Y1).*1e8; % this one can be tuned


%rho = 2.5;          % this one can be tuned
eta_Q = a*norm(Q,2).^2;%2;%./1e5;%./7.7;%./3.5;%7;
eta_WU = a*norm(W,2).^2;%2;%./1e5;%./7.7;%./3.5;%7;
eta_WQ = a*norm(W*Q,2).^2;%./1;%./7.7;%./3.5;%7;
eta_QU = a*norm(Q,2).^2;%2;%1e5;%./7.7;%./3.5;%7;

% other parameters
iter = 0;
converge = false;
norm_L = norm(L,'fro');

while (~converge && iter < maxIter)
    iter = iter + 1;
    
    % update U
    Uk1 = -(D*X-U)./mu;
    Uk2 = -Q'*(gamma1.*Y1./mu+(H-Q*U));
    Uk3 = -(W*Q)'*(gamma2.*Y2./mu+(L-W*Q*U));
    U_temp = U - (Uk1 + Uk2 + Uk3)./(eta_Q+eta_WQ);
    U = ALst(U_temp, theta./(mu*(eta_Q+eta_WQ))); 
    
    %update Q
    Q_temp = -(gamma1.*Y1./mu+(H-Q*U))*U'-W'*(gamma2.*Y2./mu+(L-W*Q*U))*U'+delta1.*Q;
    Q = Q - Q_temp./(eta_Q+eta_WU);
    
    %update W
    W_temp = -(gamma2.*Y2./mu + (L-W*Q*U))*(Q*U)'+delta2.*W;
    W = W - W_temp./eta_QU;
    
    %update D
    D=U*X'*inv(X*X'+eye(size(X,1))*lambda3);
   
    % update Y 
    C1 = H-Q*U;
    Y1 = Y1 + mu*C1;
    
    C2 = L-W*Q*U;
    Y2 = Y2 + mu*C2;

    
    % stop criterion   
    T = (norm(C1,'fro')+norm(C2,'fro'))/norm_L;
    converge = T<tol;
    
end

end