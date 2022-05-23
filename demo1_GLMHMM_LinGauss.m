% Test simulation & fitting of a GLM-HMM with linear-Gaussian outputs
%
% Model summary: 
% ---------------
% z_{t+1} | z_t = A(z_t,:)  
% y | x, z = x * w_{z} + eps,  eps ~ N(0, diag(sig_1,...,sig_nY))
%    where
% z = discrete latent variable that evolves according to an HMM
% x = set of inputs or regressors
% y = state-dependent linear transformation of x with indep Gaussian noise 

% add directories
addpath inference;
addpath models;
addpath tools; 

%% 1. Generate simulated dataset

% Set parameters: transition matrix and emission matrix
nStates = 3; % number of states
nX = 5;  % number of input dimensions (i.e., dimensions of regressor)
nY = 2;  % number of output dimensions 
nT = 1e3; % number of time bins
loglifun = @logli_LinGauss;  % log-likelihood function

% Set transition matrix by sampling from Dirichlet distr
alpha_diag = 25;  % concentration param added to diagonal (higher makes more diagonal-dominant)
alpha_full = 1;  % concentration param for other entries (higher makes more uniform)
G = gamrnd(alpha_full*ones(nStates) + alpha_diag*eye(nStates),1); % sample gamma random variables
mmtrue.A = G./repmat(sum(G,2),1,nStates); % normalize so rows sum to 1

% Set linear weights & output noise variances
mmtrue.wts = randn(nY,nX,nStates); % linear regression weights
mmtrue.vars = rand(nY,nStates)*1+.1; % variances of indep output noise

% Generate inputs (or regressors)
xx = randn(nX,nT); 

% Simulate outputs from true model using inputs xx
[yy,zlatent] = genSimGLMHMMdata_LinGauss(mmtrue,xx);

% Report fraction of time in each state
fprintf('\nState occupancies:\n');
fprintf('--------------\n');
for jj = 1:nStates
    fprintf('State %d: %.1f%%\n',jj,sum(zlatent==jj)/nT*100);
end
fprintf('--------------\n\n');

% --- Plot params and data ----------
clf; 
subplot(331);
imagesc(mmtrue.A); axis image; 
xlabel('state at time t+1'); ylabel('state at time t');
set(gca,'xtick',1:nStates,'ytick',1:nStates);
title('transition matrix A'); colorbar;

subplot(3,3,2:3);
plot(1:nX*nY, reshape(mmtrue.wts,nY*nX,nStates));
hold off; box off;
title('per-state weights');
statelabels = {'state 1', 'state 2','state 3','state 4','state 5'};
legend(statelabels{1:nStates});
xlabel('coefficient #'); ylabel('weight');

subplot(3,1,3); % plot latent state probabilities given true params
tt = 1:min(100,nT); % number of time bins to plot
plot(tt, zlatent(tt), 'k'); box off;
title('true latent state z(t)'); ylabel('state');
xlabel('trial #');
% ----------------------------------------

%% 2. Fit 1-state model

what1 = ((xx*xx')\(xx*yy'))';  % OLS estimate for weights
rr = (yy-what1*xx); % residuals
varhat1 = var(rr,0,2);  % ML estimate for variances

% Compute log-likelihood directly
logli0 = sum(sum(log(normpdf(rr,0,sqrt(repmat(varhat1,1,nT))))));
fprintf('Log-likelihood, 1-state model: %.2f\n',logli0);

% % ----------------------------------------------------------
% % Unit test: check that log-li computed with FB code agrees
% % ----------------------------------------------------------
% mm0 = mmtrue; mm0.A = 1; mm0.wts = what1; mm0.vars = varhat1; % set params
% mm0.loglifun = loglifun; % set log-li func
% logli0b = runFB_GLMHMM(mm0,xx,yy); % evaluate log-li
% fprintf('Log-likelihood (unit test):    %.2f\n',logli0b);
% % ----------------------------------------------------------


%% 3. Run forward backward and M step starting from true params

% Set loglikelihood & M-step functions
mmtrue.loglifun = loglifun;

% Run forward-backward w/ true params to get latent state probabilities 
[logpTrue,gamsTrue] = runFB_GLMHMM(mmtrue,xx,yy);

fprintf('Log-likelihood, true pparams:  %.2f\n',logpTrue);

% Plot inferred states (using true params)
subplot(313);
plot(tt,gamsTrue(:,tt)');
hold off; box off; legend(statelabels{1:nStates});
title('inferred latent state');
ylabel('P( z(t) | y(t) )'); xlabel('trial #');

if logpTrue < logli0
    fprintf('WARNING: degenerate dataset; 1-state model more likely than true model\n');
end


%% 4. Run EM to estimate model params from a random initialization

% Set EM optimization params
optsEM.maxiter = 500;  % max # of EM iterations
optsEM.dlogptol = 1e-3; % stop when change in log-likelihood falls below this
optsEM.display = 10;

% Initialize transition matrix A
A0 = 1*eye(nStates)+.1*rand(nStates)+.05;
A0 = A0 ./ sum(A0,2); % normalize rows to sum to 1

% Initialize Gaussian params (mean and var)
wts0 = what1 + .1*randn(nY,nX,nStates);  % initial means
vars0 = ones(nY,nStates)*10;      % initial variances

% Build struct for initial params
mm0 = struct('A',A0,'wts',wts0,'vars',vars0,...
    'loglifun',loglifun,'Mstepfun',@runMstep_LinGauss);

% --- run EM -------
fprintf('\n-----------\nRunning EM...\n-----------\n');
[mmhat,logp,logpTrace,jStop,dlogp,gams1] = runEMforGLMHMM(mm0,xx,yy,optsEM);

% Check EM stopping conditions 
if jStop==optsEM.maxiter
    fprintf('EM terminated after max # iters (%d) reached (dlogp = %.4f)\n',jStop,dlogp);
else
    fprintf('EM terminated after %d iters (dlogp = %.4f)\n',jStop,dlogp);
end
fprintf('\nrelative final log-likelihood: %.2f \n', logp-logpTrue);
if logp>logpTrue, fprintf('(SUCCESS!)\n');
else,   fprintf('(FAILED to find optimum!)\n');
end

% ---- Plot log-li vs EM iterations ------------------
subplot(324);  
plot([1 min(jStop+20,optsEM.maxiter)],logpTrue*[1 1], 'k',...
    1:jStop,logpTrace(1:jStop),'-');
xlabel('EM iteration');
ylabel('log p(Y|theta)');
title('EM path');


%% 5. Display results

% Permute states to find best match to true model states
wtshatMat = reshape(mmhat.wts,[],nStates);  % matrix of reshaped weights 
Mperm = computeAlignmentPermutation(reshape(mmtrue.wts,[],nStates),wtshatMat); % find permuatation matrix
Ahat = Mperm*mmhat.A*Mperm'; % permute transition matrix 
wtshatPerm = wtshatMat*Mperm'; % permuted weights
wtshat = reshape(wtshatPerm,nY,nX,nStates); % reshaped permuted weights
varhat = mmhat.vars*Mperm'; % permuted variances
gams = Mperm*gams1; % permute the posterior latent probabilities

% ---- make plots -------

% set legend labels & colors
labelsInf = {'inferred 1', 'inferred 2','inferred 3', 'inferred 4', 'inferred 5'};
hcol1 = get(gca,'colororder')*0.66;  % colors for inferred states & wts

subplot(3,3,2:3);
plot(1:nX*nY, reshape(mmtrue.wts,nY*nX,nStates)); 
hold on; % plot estimates
for jj = 1:nStates, plot(1:nX*nY,wtshatPerm(:,jj),'--','color',hcol1(jj,:)); end
hold off;
title('per-state weights & estimates');
legend({statelabels{1:nStates},labelsInf{1:nStates}}); hold off; box off;

subplot(323);  % plot variances
plot(1:nY*nStates,mmtrue.vars(:),'ko',1:nY*nStates,varhat(:),'r*');
xlabel('output #');  ylabel('variance');
legend('true', 'estimate');
title('output noise variances');

subplot(3,1,3); % plot latent states inferred
plot(tt,gamsTrue(:,tt)');
hold off; box off; 
title('inferred latent state');
ylabel('P( z(t) | y(t) )'); xlabel('trial #'); hold on;
for jj = 1:nStates, plot(tt,gams(jj,tt)','--','color',hcol1(jj,:)); end
hold off;
legend({statelabels{1:nStates},labelsInf{1:nStates}}); hold off;
