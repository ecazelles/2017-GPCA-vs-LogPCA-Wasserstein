function [v_gpca, t_gpca,residual] = algo_GPCA_1D_surface(w,Omega,L,w0,f,iter_max,iter_sub_max)
% Function:
%      Compute principal geodesics via Geodesic surface approach in [Geodesic PCA versus Log-PCA of histograms in the Wasserstein
%      space. E. Cazelles, V. Seguy and al.]
%
% Usage:
%       [v_gpca, t_gpca,residual] = algo_GPCA_1D_surface(w,Omega,L,w0,f)
%   input:
%       w    = log maps of the data at the barycenter f
%       Omega     = support of the data
%       L   = number of component to estimate
%       w0   = initialization of the components
%       f = Wasserstein barycenter of the data
%       iter_max = Maximum number of iterations
%       iter_sub_max = Maximum number of iterations for the proximity operator
%
%   output:
%       v_gpca  = principal component
%       t_gpca  = position of the projection of the data onto principal
%       components
%       residual = residual error
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

if nargin<6
    iter_max=10000;
end

if nargin<7
    iter_sub_max=100;
end

[m n]=size(w);  % w contain m data of dimension n

[M n]=size(w0);  % contain M initialization vectors

t_gpca = zeros(m,L);

a = Omega(1);
b = Omega(end);

dis=Omega(2)-Omega(1);

beta=0.2;

% Gradient descent steps
taut=1e-5;
tauv=0.001;

%Primal-dual parameters:
delta = 2/dis;
sigma=1/delta;
theta=tauv/(1+delta*tauv);



%Parameter for approximation of strictly increasing functions
epsilon=1e-2;

%Unknown to estimate
v_gpca=zeros(L,n);

threshold=1e-20;
threshold_sub=1e-20;

v=w0(1:L,:);
    

%Initalize t w.r.t v
vr =bsxfun(@times,v,sqrt(f));
taux=zeros(m,L);
for l=1:L
    taux(:,l)=(sum(w.*repmat((f.*v(l,:)), [m 1]),2)/sqrt(sum(vr(l,:).*vr(l,:))));
end
t_pos = taux;
t_neg = -taux;

vm1=1.;
z=v*0;
iter=1;

t_old_pos=t_pos;
t_old_neg=t_neg;
v_old=v;

minu = max((a-Omega),-(b-Omega));
maxu = min(-(a-Omega),(b-Omega));


while iter<=iter_max && sum(abs(v(:)-vm1(:)))/sum(abs(vm1(:)))>threshold

    iter=iter+1;

    vm1=v;
    tm1_pos=t_pos;
    tm1_neg=t_neg;

    t_pos=t_pos+beta*(t_pos-t_old_pos)-taut*((t_pos-t_neg)*v-w)*(repmat(f,[L,1]).*v)';

    t_neg=t_neg+beta*(t_neg-t_old_neg)-taut*(-(t_pos-t_neg)*v+w)*(repmat(f,[L,1]).*v)';
    
    for i=1:m
        y=projsplx([t_pos(i,:) t_neg(i,:)]);
        t_pos(i,:)=y(1:l);
        t_neg(i,:)=y(l+1:end);
    end

    v=v+beta*(v-v_old)-tauv*bsxfun(@times,f,((t_pos-t_neg)'*((t_pos-t_neg)*v-w)));
            

            
    %computation of proximity operator
    um1=1.;
    u=v;  %primal variable for prox
    ut=v; %auxiliary primal variable
    iter_sub=1;
    %Primal-dual algorithm
    while ( (iter_sub<=iter_sub_max) && (sum(abs(u(:)-um1(:)))/sum(abs(um1(:)))>threshold_sub) )

        iter_sub=iter_sub+1;
        for l=1:L
        %Dual ascent
            z(l,1:end-1)=z(l,1:end-1)+sigma*((ut(l,2:end)-ut(l,1:end-1))/dis);

            %Dual projection
            II=z(l,:)<sigma*(1-epsilon) & z(l,:)>-sigma*(1-epsilon);
            z(l,II)=0;
            II=z(l,:)>sigma*(1-epsilon);
            z(l,II)=z(l,II)-sigma*(1-epsilon);
            II=z(l,:)<-sigma*(1-epsilon);
            z(l,II)=z(l,II)+sigma*(1-epsilon);
        end

        ut=u;
        um1=u;

        for l=1:L
            div=u(l,:)*0;
            div(1)=z(l,1)/dis;
            div(2:end-1)=(z(l,2:end-1)-z(l,1:end-2))/dis;
            div(end)=-z(end-1)/dis;
            %Primal descent
            u(l,:)=u(l,:)-theta*((u(l,:)-v(l,:))/tauv-div);        
            %Primal projection
            u(l,:)=min(maxu,max(minu,u(l,:)));
        end

        %auxiliary variable update
        ut=2*u-ut;

    end

    v=u;   %u is the value of the proximity operator

    t_old_pos=tm1_pos;
    t_old_neg=tm1_neg;
    v_old=vm1;


end

v_gpca=v;
t_gpca=t_pos-t_neg;

% Normalization
A = v_gpca * diag(f) * v_gpca';
for l=1:L
    v_res(l,:) = v_gpca(l,:) / sqrt(A(l,l));
    t_res(:,l)=sqrt(A(l,l))*t_gpca(:,l);
end

% Residual
Vp_g = zeros(size(w,1),length(Omega));
for i = 1:size(w,1)
    Vp_g(i,:)=t_res(i,:)*v_res;
end

residual = sum(sum((bsxfun(@times,(w-Vp_g).^2,f)),2),1);
