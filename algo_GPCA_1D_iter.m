function [v_gpca_opt, t_gpca_opt,t0_opt,residual_opt,W_residual_g] = algo_GPCA_1D_iter(w,Omega,L,w0,f,range_t0,iter_max,iter_sub_max)
% Function:
%      Compute principal geodesics via Iterative Geodesic approach in [Geodesic PCA versus Log-PCA of histograms in the Wasserstein
%      space. E. Cazelles, V. Seguy and al.]
%
% Usage:
%       [v_gpca_opt, t_gpca_opt,t0_opt,residual_opt,W_residual_g] = algo_GPCA_1D_iter(w,Omega,L,w0,f,range_t0)
%   input:
%       w    = log maps of the data at the barycenter f
%       Omega     = support of the data
%       L   = number of component to estimate
%       w0   = initialization of the components
%       f = Wasserstein barycenter of the data
%       range_t0 = values taken by the centering variable t0
%       iter_max = Maximum number of iterations
%       iter_sub_max = Maximum number of iterations for the proximity operator
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

if nargin<7
    iter_max=10000;
end

if nargin<8
    iter_sub_max=100;
end

[m n]=size(w);  %w contain m data of dimension n

[M n]=size(w0);  % contain M initialization vectors

wc=mean(w,1);

t_gpca = zeros(m,L);
t_gpca_opt = zeros(m,L);

a = Omega(1);
b = Omega(end);

d=Omega(2)-Omega(1);

beta=0.2;

% Gradient descent steps
taut=1e-5;
tauv=0.01;

%Primal-dual parameters:
delta = 2/d;
sigma=1/delta;

theta=tauv/(1+delta*tauv);



%Parameter for approximation of strictly increasing functions
 epsilon=1e-1;

%Unknown to estimate
v_gpca=zeros(L,n);
v_gpca_opt = zeros(L,n);
v_gpca_opt_res = zeros(L,n);

threshold=1e-8;
threshold_sub=1e-6;

nb_t0=length(range_t0);
W_residual_g=zeros(nb_t0,L);
residual_opt=1e15*ones(1,L);
t0_opt=zeros(1,L);


for l=1:L  %estimation of the component l=1...L
    
    display(['Compute component ' num2str(l) '/' num2str(L)]);

    for tt=1:nb_t0
        t0=range_t0(tt);
        minu = max((a-Omega)/(t0+1),(b-Omega)/(t0-1));
        maxu = min((a-Omega)/(t0-1),(b-Omega)/(t0+1));

        if l<= M  %initalize v with w0
            v=w0(l,:);
        end

        %Initalize t w.r.t v
        vr = sqrt(f).*v;
        taux = (sum(w.*repmat((f.*v), [m 1]),2)/sqrt(sum(vr(:).*vr(:))));

        t = taux;


        vm1=1.;
        z=v*0; %dual variable
        iter=1;

        t_old=t;
        v_old=v;



        while iter<=iter_max && sum(abs(v(:)-vm1(:)))/sum(abs(vm1(:)))>threshold

            iter=iter+1;

            vm1=v;
            tm1=t;
                                                            
            t=min(1,max(-1,t+beta*(t-t_old)-taut*sum(repmat((f.*v),[m 1]).*((t+t0)*v-w),2)));
            v=v+beta*(v-v_old)-tauv*f.*sum(repmat(tm1+t0,[1 n]).*((tm1+t0)*v-w),1);  
            
            
           % Computation of proximity operator
            um1=1.;
            u=v;  %primal variable for prox
            ut=v; %auxiliary primal variable

            iter_sub=1;
            %Primal-dual algorithm
            while ( (iter_sub<=iter_sub_max) && (sum(abs(u(:)-um1(:)))/sum(abs(um1(:)))>threshold_sub) )

                iter_sub=iter_sub+1;
                %Dual ascent
                z(1:end-1)=z(1:end-1)+sigma*(ut(2:end)-ut(1:end-1))/d;

                %Dual projection
                II=z<sigma*(1-epsilon)/(1-t0)& z>-sigma*(1-epsilon)/(1+t0);
                z(II)=0;
                II=z>sigma*(1-epsilon)/(1-t0);
                z(II)=z(II)-sigma*(1-epsilon)/(1-t0);
                II=z<-sigma*(1-epsilon)/(1+t0);
                z(II)=z(II)+sigma*(1-epsilon)/(1+t0);

                ut=u;
                um1=u;

                div=u*0;
                div(1)=z(1);
                div(2:end-1)=z(2:end-1)-z(1:end-2);
                div(end)=-z(end-1);
                %Primal descent
                u=u-theta*((u-v)/tauv-div/d);


                %Primal projection
                aux = 0;
                for k=1:l-1
                    v_gpca_r = sqrt(f).*v_gpca(k,:);
                    aux = aux + sum(f.*v_gpca(k,:).*u)/sum(v_gpca_r.*v_gpca_r)*v_gpca(k,:);
                end
                u = u - aux;
                u=min(maxu,max(minu,u));

                ut=2*u-ut;

            end
            
            v=u;   %u is the value of the proximity operator

            t_old=tm1;
            v_old=vm1;



        end
        iter;
        v_gpca(l,:)=v;
        t_gpca(:,l) = t+t0;

  
        % Normalization
         A = v_gpca(l,:) * diag(f) * v_gpca(l,:)';
         v_gpca(l,:) = v_gpca(l,:) / sqrt(A);
         t_gpca(:,l)=sqrt(A)*(t_gpca(:,l));
        
         Vp_g = zeros(size(w,1),length(Omega));
        
        % Residual
        for i = 1:size(w,1)
            Vp_g(i,:)=zeros(1,length(Omega));
            for lp = 1:l
                Vp_g(i,:) =t_gpca(i,lp)*v_gpca(lp,:)+Vp_g(i,:);
            end
            W2(i)=sum(((w(i,:)-Vp_g(i,:)).^2).*f);
        end
        
        W_residual_g(tt,l)=sum(W2);
        
        if W_residual_g(tt,l)<residual_opt(l)
            residual_opt(l)=W_residual_g(tt,l);
            t0_opt(l)=t0;
            v_gpca_opt(l,:)=v_gpca(l,:);
            t_gpca_opt(:,l)=t_gpca(:,l);
        end
        
      end
     
end
