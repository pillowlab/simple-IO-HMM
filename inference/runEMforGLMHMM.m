function [mmhat,logp,logpTrace,jj,dlogp,gams] = runEMforGLMHMM(mm0,xx,yy,optsEM,mask)
% [mmhat,logp,logpTrace,jj,dlogp,gams] = runEMforGLMHMM(mm0,xx,yy,optsEM,mask)
%
% Run EM for GLM-HMM with Gaussian observations
%
% Inputs:
% -------
%   mm0 [struct] - struct with initial params
%        .A0    [k x k] - initial state transition matrix
%        .mus0  [1 x k] - initial mean for each state
%        .vars0 [1 x k] - initial variance for each state
%    rr [1 x T] - residual for each time bin
%  opts [struct] - optimization params (optional)
%       .maxiter - maximum # of iterations 
%       .dlogptol - stopping tol for change in log-likelihood 
%       .display - how often to report log-li
%
% Outputs:
% --------
%     mmhat [struct] - struct with fitted params.  Fields:
%            .A    [k x k] - estimated state transition matrix
%            .mus  [1 x k] - estimated per-state means
%            .vars [1 x k] - estimated per-state variances
%      logp [1 x 1] - log-likelihood
% logpTrace [1 x maxiter] - trace of log-likelihood during EM
%        jj [1 x 1] - final iteration
%     dlogp [1 x 1] - final change in log-likelihood
%      gams [k x T] - marginal state probabilities from last E step


% Set EM optimization params if necessary
if nargin < 4
    optsEM.maxiter = 200;
    optsEM.dlogptol = 0.01;
    optsEM.display = inf;
end
if ~isfield(optsEM,'display') || isempty(optsEM.display)
    optsEM.display = inf;
end

% Create mask if necessary
if nargin < 5
    mask = true(1,length(yy));
end

% Initialize params
mmhat = mm0; 

% Set up variables for EM
logpTrace = zeros(optsEM.maxiter,1); % trace of log-likelihood
dlogp = inf; % change in log-likelihood
logpPrev = -inf; % prev value of log-likelihood
jj = 1; % counter

while (jj <= optsEM.maxiter) && (dlogp>optsEM.dlogptol)
    
    % --- run E step  -------
    [logp,gams,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask); % run forward-backward
    logpTrace(jj) = logp;
   
    % --- run M step  -------
   
    % Update transition matrix
    mmhat.A = xisum ./ sum(xisum,2); % normalize each row to sum to 1
    
    % Update model params
    mmhat = mmhat.Mstepfun(mmhat,xx(:,mask),yy(:,mask),gams(:,mask));
    
    % ---  Display progress ----
    if mod(jj,optsEM.display)==0
        fprintf('EM iter %d:  logli = %-.6g\n',jj,logp);
    end
    
    % Update counter and log-likelihood change
    jj = jj+1;  
    dlogp = logp-logpPrev; % change in log-likelihood
    logpPrev = logp; % previous log-likelihood

    if dlogp<-1e-6
        warning('Log-likelihood decreased during EM!');
        fprintf('dlogp = %.5g\n', dlogp);
    end

end
jj = jj-1;

