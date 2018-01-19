function h = pushforward_density(T,f,OmegaExt)
% Function:
%      Compute the pushforward density of a histogram by a map non-
%      necessary increasing
%
% Usage:
%       h = pushforward_density(T,f,OmegaExt)
%   input:
%       T    = the map used to push the histogram
%       f     = the histogram to transport
%       OmegaExt   = the support of the histogram f to transport
%
%   output:
%       hfinal   = the histogram transported by the map T from the measure
%       f
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


epsilon = 1e-5;
int=OmegaExt(2)-OmegaExt(1);

% Divide the support OmegaExt into intervals for which T is monotonic
cst=find(abs(T(1:end-1)-T(2:end))<epsilon);
a=SplitVec(cst,'consecutive');

noncst=1:length(OmegaExt);
noncst(cst)=[];
b=SplitVec(noncst,'consecutive');

dR = -savitzkyGolayFilt(T(noncst),1,1,15)/int;
g=interp1(T(noncst),f(noncst)./abs(dR),OmegaExt(noncst),'pchip');

h=zeros(1,length(OmegaExt));
h(noncst)=g;


if isempty(a)
    return
else
    if a{1}(1)==1
       h(a{1})=zeros(1,length(a{1}));
    else
        for i=1:length(a)
            h(a{i})=h(a{i}(1)-1);
        end
    end
end
