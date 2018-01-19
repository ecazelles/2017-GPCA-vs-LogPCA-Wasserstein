function V = logMap(h,Fb,S,method)
% Function:
%      Compute the logarithmic map of an one-dimensional histogram at a reference measure
%
% Usage:
%       h   = an one-dimensional histogram
%       Fb  = the histogram of the reference measure
%       S   = support of the histogram
%       method   = smoothing method used among 'pchip', 'linear' and 'spline'
%
%   output:
%       Finv   = smooth quantile function of the input histogram
%       barycenter
% Example:
%       logMap(h,Fb,S,'spline')
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
A = (h > epsilon);
A(end) = 0;
Sf = S(A);

H = cumsum(h);
Hf = H(A);

if (Hf(end) == 1)
	Hf(end) = Hf(end) - eps;
end

if strcmp(method,'pchip')

    FFinv = pchip([0 Hf 1],[S(1)-1 Sf S(end)], Fb);

elseif strcmp(method,'spline')
    
    FFinv = spline([0 Hf 1],[S(1)-1 Sf S(end)], Fb);
    
elseif strcmp(method,'linear')
     
    FFinv = interp1([0 Hf 1],[S(1)-1 Sf S(end)], Fb);
        
end

d=S(2)-S(1);

V = FFinv - [S(1)-d S];
