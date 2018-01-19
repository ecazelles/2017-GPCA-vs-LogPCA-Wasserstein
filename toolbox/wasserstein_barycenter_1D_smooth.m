function [B, FB] = wasserstein_barycenter_1D_smooth(mu,S,method,n_inv)
% Function:
%      Compute a smooth Wasserstein barycenter of one-dimensional histograms
%
% Usage:
%       [B,FB] = wasserstein_barycenter_1D_smooth(mu,S,method,n_inv)
%   input:
%       mu    = matrix of histograms * support
%       S     = support of the histograms
%       method   = smoothing method used among 'pchip', 'linear' and 'spline'
%       n_inv   = number of support points for quantile functions of the
%       histograms
%
%   output:
%       B   = histogram of the cumulative distribution function of the
%       barycenter
%       FB   = histogram of the density of the barycenter
% Example:
%       wasserstein_barycenter_1D_smooth(mu,S,'spline',10000)
%       wasserstein_barycenter_1D_smooth(mu,S,'linear',5000)
% Authors:
%       Elsa Cazelles, Institut de Mathématiques de Bordeaux, Université
%       Bordeaux.
%       Vivien Seguy, Graduate School of Informatics, Kyoto University.
%       Jérémie Bigot, Institut de Mathématiques de Bordeaux, Université
%       Bordeaux.
%       Marco Cuturi, CREST, ENSAE, Université de Paris Saclay.
%       Nicolas Papadakis, Institut de Mathématiques de Bordeaux, CNRS.
%
% Copyright 2017 Elsa Cazelles, Vivien Seguy


Sinv = 0:1/n_inv:1;
Binv = zeros(1,length(Sinv));
N = size(mu,1);

for i = 1:size(mu,1)
    Binv = Binv +  histogram_pseudo_inverse_smooth(mu(i,:),S,Sinv,method);
end
Binv = Binv/N;

FB = pchip(Binv,Sinv,[S(1)-1 S]);
B = diff(FB);

end
