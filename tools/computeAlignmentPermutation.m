function Mperm = computeAlignmentPermutation(A1, A2)
% Mperm = computeAlignmentPermutation(P1, P2)
%
% Compute a permutation matrix to align columns of A2 with A1, such that
% A2*Mperm' is as close to A1 as possible.

nStates = size(A1,2);
pCorr = A1'*A2; % correlation of true and inferred Phis
Mperm = zeros(nStates); % permutation matrix
for jj = 1:nStates 
    [~,ind] = max(pCorr(:)); % find cols with max correlation
    [i,j] = ind2sub([nStates,nStates],ind);
    Mperm(i,j) = 1; % insert a 1 into the permutation matrix
    pCorr(i,:) = -inf; % remove this row from phiCorr
    pCorr(:,j) = -inf; % remove this column from phiCorr
end

