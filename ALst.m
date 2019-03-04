% function to process soft thresholding of a matrix
% Input: E, the input matrix
%        beta, the soft thresholding parameter
% Output: E_hat, the output matrix
% 04-05-2013, Bian Xiao

function E_hat = ALst(E, beta)

% E_hat = max(E-beta,0) + min(E+beta,0);
E_hat = sign(E).*max(abs(E)-beta, 0);