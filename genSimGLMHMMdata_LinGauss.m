function [yresp, zlatent] = genSimGLMHMMdata_LinGauss(mm,xstim)
% [yresp, zlatent] = genSimGLMHMMdata_Gaussian(mm,xstim)
%
% Simulate data from a GLM-HMM with linear Gaussian observations
%
% Inputs:
% ------
%     mm [struct] - model struct with fields:
%           .A [k,k]   - transition matrix (k states)
%           .wts [d,k] - GLM filters for each state
%           .vars[1,k] - variance of output noise for each state 
%
%  xstim [nx x T] - design matrix (columns are input vectors per time bin)
%
% Output:
% ------
%    yobs [nY x T] - observation at each time bin
% zlatent [1 x T]  - latent state at each time bin

nStates = size(mm.A,1); % number of states
nT = size(xstim,2); % number of time bins
nY = size(mm.wts,1); % number of output dimensions
stds = sqrt(mm.vars); % stdev of Gaussian noise for each output & state

% allocate space for latents and response
zlatent = zeros(1,nT); % latents
yresp = zeros(nY,nT); % observations

% Sample first state and observation
zz = randsample(nStates,1); % sample first state from uniform distribution

% Sample data from the model
for jj = 1:nT
    zlatent(jj) = zz; % store latent
    yresp(:,jj) = mm.wts(:,:,zz)*xstim(:,jj) + stds(:,zz).*randn(nY,1); % sample observation
    zz = randsample(nStates,1,true,mm.A(zz,:)); % sample next latent state
end

