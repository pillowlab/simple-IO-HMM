function logpy = logli_LinGauss(mm,xx,yy)
% logpy = logli_Gaussian(mm,xx,yy)
%
% Compute log-likelihood term under Gaussian observation model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%      .mus  [1 K] - per-state means
%      .vars [1 K] - per-state variances
%    xx [1 T] - inputs
%    yy [1 T] - outputs
%
% Output
% ------
%  logpy [T K] - loglikelihood for each observation for each state

% Compute linear prediction for each state
yypred = pagemtimes(mm.wts,xx);  % linear prediction
rr = yy-yypred;  % residuals

% log P( Y | X, theta)
vartensor = 2*permute(mm.vars,[1,3,2]);  % reshaped variances
logpy_all = -(rr).^2./vartensor - 0.5*log(pi*vartensor); % log-li of each output at each time for each state
logpy = permute(sum(logpy_all,1),[2,3,1]); % log-li per state at each time
