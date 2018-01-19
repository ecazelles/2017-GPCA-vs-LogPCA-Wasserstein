function Finv = histogram_pseudo_inverse_smooth(mu,S,Sinv,method)
% Function:
%      Compute a smooth quantile function of an one-dimensional histogram
%
% Usage:
%       Finv = histogram_pseudo_inverse_smooth(mu,S,Sinv,method)
%   input:
%       mu    = an one-dimensional histogram
%       S     = support of the histogram
%       Sinv   = support points for quantile function of the histogram
%       method   = smoothing method used among 'pchip', 'linear' and 'spline'
%
%   output:
%       Finv   = smooth quantile function of the input histogram
%       barycenter
% Example:
%       histogram_pseudo_inverse_smooth(mu,S,'Sinv','spline')
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

epsilon = 10^-15;
A = (mu > epsilon);
A(end) = 0;
% muf = mu(A);
Sf = S(A);

mua = cumsum(mu);
mufa = mua(A);
if (mufa(end) == 1)
        mufa(end) = mufa(end) - eps;
end
    
if strcmp(method,'pchip')
    
    Finv = pchip([0 mufa 1],[S(1)-1 Sf S(end)],Sinv);

elseif strcmp(method,'spline')
    
    Finv = spline([0 mufa 1],[S(1)-1 Sf S(end)],Sinv);
    
elseif strcmp(method,'linear')
     
    Finv = interp1([0 mufa 1],[S(1)-1 Sf S(end)],Sinv,'linear',0);
        
end

