function mm = runMstep_LinGauss(mm,xx,yy,gams)
% mm = runMstep_LinGauss(mm,xx,yy,gams)
%
% Run m-step updates for Gaussian observation model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%        .wts  [1 K] - per-state slopes
%        .vars [1 K] - per-state variances
%    xx [d T] - input (design matrix)
%    yy [1 T] - outputs
%  gams [K T] - log marginal sate probabilities given data
%
% Output
% ------
%  mmnew - new model struct

% Determine which params to update:
updatewts = ~isfield(mm,'updatewts') || mm.updatewts;
updatevars = ~isfield(mm,'updatevars') || mm.updatevars;

% normalize the gammas to sum to 1
gamnrm = gams./(sum(gams,2)+1e-8);  
nStates = size(gams,1);


if updatewts  % update weights (if desired)
    for jj = 1:nStates
        xxw = xx.*gamnrm(jj,:);  % weighted inputs;
        mm.wts(:,:,jj) = ((xxw*xx')\(xxw*yy'))'; % weighted least-squares estimate
    end
end

% Update variances (if desired)
if updatevars

    % Compute linear prediction for each state
    yypred = pagemtimes(mm.wts,xx);  % linear prediction
    rr = yy-yypred;  % residuals

    % update variances
    mm.vars = permute(sum((rr.^2).*permute(gamnrm,[3 2 1]),2),[1 3 2]);
end

