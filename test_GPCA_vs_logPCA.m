%%
% Test for the computation of both log-PCA and GPCA (applied to gaussian
% measures)
%
% Matlab code to reproduce the results of the paper
% 'Geodesic PCA versus Log-PCA of histograms in the Wasserstein space', E. Cazelles, V. Seguy, J. Bigot, M. Cuturi, N. Papadakis
% Arxiv: https://arxiv.org/abs/1708.08143
%
% Copyright 2017 Elsa Cazelles, Vivien Seguy

addpath('toolbox')

clear all
close all

%% Generate synthetic gaussian data

display('Gaussian data')
% Support of the data
Omega = -200:200;
OmegaExt = [Omega(1)-1 Omega]; % we will define the density in Omega(1)-1 as equal to zero

% Number of observations (ie histograms)
n = 50;

% Parameters of gaussian histograms
param = zeros(n,2);
param(:,1) = unifrnd(-120,120,n,1);
param(:,2) = unifrnd(1,4,n,1);

m = 0;
sigma = 8;

% Matrix of the data
mu = zeros(n,length(Omega));
for j = 1:n
   mu(j,:) = normpdf(Omega,m+param(j,1),sigma*param(j,2));
   mu(j,:) = mu(j,:)/sum(mu(j,:));
end

figure(1)
plot(Omega,mu)
title('Data');

% True Wasserstein barycenter (closed form for gaussian measures)
display('True Wasserstein barycenter')
figure(2)
moy_tb=mean(m+param(:,1));
s_tb=mean(sigma*param(:,2));
tb=normpdf(Omega,moy_tb,s_tb);
plot(Omega,tb);
title('True Wasserstein barycenter')

%% Compute euclidean PCA on data

display('Euclidean PCA')
Dc = mean(mu,1);
Dp = bsxfun(@minus,mu,Dc); % centering the data
D = Dp'*Dp/size(Dp,1); % covariance matrix
[eigD, eigValsD] = eig(D); % perform euclidean PCA

eigD = eigD(:,end:-1:1); % descending order is simpler for notations
eigValsD = diag(eigValsD); 
eigValsD = eigValsD(end:-1:1);
tD = bsxfun(@minus,mu,Dc)*eigD; % inner product of the dataset with the eigenvectors

l = 1; % choose the principal component index (1<= l <= length(Omega))
[~, ID] = sort(tD(:,l)); % ordering the data

gD = zeros(size(mu,1),length(Omega));    

% Projection of the data along the l-th principal component in Euclidean
% PCA
figure(3)
for i = 1:size(mu,1)
    gD(i,:) = Dc+tD(ID(i),l)*eigD(:,l)';
    plot(Omega,[gD(i,:) ; mu(ID(i),:)]);
    legend(['Projection of the data onto the PC ',num2str(l)],'Data')
    title('Data projection along principal component in Euclidean PCA');
    axis([Omega(1) Omega(end) 0 0.2]);
    pause(0.02);
end

% Reprensentation of the 1st and 2nd components of the Euclidean PCA
figure(4);
map1=autumn(size(mu,1));
map2=cool(size(mu,1));

subaxis(1,2,1,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
[~, I] = sort(tD(:,1));
for i = 1:size(mu,1)
    gD(i,:) = Dc+tD(I(i),1)*eigD(:,1)';
    plot(Omega,gD(i,:),'Color',map1(i,:));
    axis([Omega(1) Omega(end) -0.008 0.022])
    hold on    
end
pl=plot(Omega,Dc,'-b','linewidth',3);
legend(pl,'Euclidean barycenter');
title('First PC','FontSize',10,'FontWeight','bold')
ylabel('Euclidean PC','FontSize',10,'FontWeight','bold');

subaxis(1,2,2,'SpacingVert',0.07,'SpacingHoriz',0.07);
[~, I] = sort(tD(:,2));
for i = 1:size(mu,1)
    gD(i,:) = Dc+tD(I(i),2)*eigD(:,2)';
    plot(Omega,gD(i,:),'Color',map1(i,:));
    axis([Omega(1) Omega(end) -0.008 0.022])
    hold on    
end
pl=plot(Omega,Dc,'-b','linewidth',3);
legend(pl,'Euclidean barycenter');
title('Second PC','FontSize',10,'FontWeight','bold')

    
%% Compute smooth barycenter

display('Smooth Wasserstein barycenter')
method = 'pchip';
n_inv = 100000; 
[Bs, FBs] = wasserstein_barycenter_1D_smooth(mu,Omega,method,n_inv);
f = [Bs(1) Bs]; % Smooth histogram of the barycenter
figure(5)
plot(Omega,Bs)
title('Smooth barycenter');

%% Compute log maps

display('Log-maps of the data at the barycenter')
method = 'pchip';
V = zeros(size(mu,1),length(Omega)+1);
for i = 1:size(mu,1)
    V(i,:) = logMap(mu(i,:),FBs,Omega,method); % log maps of the data at the barycenter
end
figure(6)
plot(OmegaExt,V)
title('Log-maps of the data at the barycenter')

%% Test move barycenter
% Computation of the exponential map via a pushforward function at the
% barycenter

display('Computation of the exponential map via a pushforward function at the barycenter')
figure(7)
for i = 1:size(V,1)
    
    T = OmegaExt+V(i,:);
    hi = pushforward_density(T,f,OmegaExt);   
    plot(OmegaExt,[hi; [mu(i,1) mu(i,:)]]);
    title('Test move barycenter');
    legend('Exponential map of the data at the barycenter','Data')
    drawnow
end

%% Compute PCA on logmap (log-PCA)

display('Log-PCA approach')

Vc = mean(V,1);
Vp = bsxfun(@minus,V,Vc)*diag(sqrt(f));
C = Vp'*Vp/size(Vp,1); % covariance matrix
[eigV, eigVals] = eig(C); % perform PCA

% Normalisation
nonzero_ind = (f > 0);
eigV(nonzero_ind,:) = diag(1./sqrt(f(nonzero_ind))) * eigV(nonzero_ind,:);

eigV = eigV(:,end:-1:1); % descending order is simpler for notations
eigVals = diag(eigVals); 
eigVals = eigVals(end:-1:1);
tV = bsxfun(@minus,V,Vc)*diag(f)*eigV; % inner product of the dataset with the eigenvectors

% Projection of the data along the l-th principal component in log-PCA
l = 1; % choose the principal component index
[~, I] = sort(tV(:,l));

hV = zeros(size(mu,1),length(OmegaExt));    
figure(8);
for i = 1:size(V,1)
    
    TV = OmegaExt+Vc+tV(I(i),l)*eigV(:,l)';
    hV(i,:) = pushforward_density(TV,f,OmegaExt);
    plot(OmegaExt,[hV(i,:) ; [mu(I(i),1) mu(I(i),:)]]);
    axis([Omega(1) Omega(end) 0 0.2]);
    legend(['Projection of the data onto the PC ',num2str(l)],'Data')
    title('Data projection along principal component in log-PCA');
    drawnow
end

% Representation of the 1st and 2nd components of log-PCA
figure(9);

subaxis(1,2,1,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
[~, I] = sort(tV(:,1));
for i=1:size(V,1)
    TV(i,:) = OmegaExt+Vc+tV(I(i),1)*eigV(:,1)';
    hV(i,:) = pushforward_density(TV(i,:),f,OmegaExt);
    plot(OmegaExt,hV(i,:),'Color',map1(i,:))
    axis([Omega(1) Omega(end) 0 0.025])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('First PC','FontSize',10,'FontWeight','bold')
ylabel('log-PC','FontSize',10,'FontWeight','bold');

subaxis(1,2,2,'SpacingVert',0.07,'SpacingHoriz',0.07);
[~, I] = sort(tV(:,2));
for i=1:size(V,1)
    TV(i,:) = OmegaExt+Vc+tV(I(i),2)*eigV(:,2)';
    hV(i,:) = pushforward_density(TV(i,:),f,OmegaExt);
    plot(OmegaExt,hV(i,:),'Color',map2(i,:))
    axis([Omega(1) Omega(end) 0 0.045])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('Second PC','FontSize',10,'FontWeight','bold')
drawnow;
%% GPCA - Iterative Geodesic Approach
 
display('Iterative Geodesic approach')

L = 2;
% Choose initialization
V0 = rand(L,length(Omega)+1); % random initialization
%V0(:,(f > 0)) = bsxfun(@rdivide,V0(:,nonzero_ind),sqrt(f(nonzero_ind))); % initialization with the eigenVector of the covariance matrix
range_t0=[-0.1:0.01:0.1];
[v_gpca_iter, t_gpca_iter,t0_iter,residual_iter,W_residual_iter] = algo_GPCA_1D_iter(V,OmegaExt,L,V0,f,range_t0);

% Representation of the 1st, 2nd components and the principal geodesic surface of
% the iterative geodesic approach
figure(10)
map3=summer(size(mu,1));
% 1st component
subaxis(1,3,1,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
h_iter = zeros(size(mu,1),length(OmegaExt));
[~, I] = sort(t_gpca_iter(:,1));
for i=1:size(mu,1)
    T_iter = OmegaExt+t_gpca_iter(I(i),1)*v_gpca_iter(1,:);
    h_iter(i,:) =pushforward_density(T_iter,f,OmegaExt);
    plot(OmegaExt,h_iter(i,:),'Color',map1(i,:))
    axis([Omega(1) Omega(end) -0.002 0.04])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
ylabel('Iterative Geodesic approach','FontSize',10,'FontWeight','bold');
title('First PG','FontSize',10,'FontWeight','bold');

% 2nd component
subaxis(1,3,2,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
[~, I] = sort(t_gpca_iter(:,2));
for i=1:size(mu,1)
    T_iter = OmegaExt+t_gpca_iter(I(i),2)*v_gpca_iter(2,:);
    h_iter(i,:) =pushforward_density(T_iter,f,OmegaExt);
    plot(OmegaExt,h_iter(i,:),'Color',map2(i,:))
    axis([Omega(50) Omega(end-50) -0.002 0.047])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('Second PG','FontSize',10,'FontWeight','bold');

% Principal Surface
subaxis(1,3,3,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
U=t_gpca_iter*v_gpca_iter;
for i=1:size(mu,1)
    T_iter(i,:) = OmegaExt+U(I(i),:);
    h_iter(i,:) =pushforward_density(T_iter(i,:),f,OmegaExt);
    plot(OmegaExt,h_iter(i,:),'Color',map3(i,:))
    axis([Omega(1) Omega(end) -0.002 0.047])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('Surface PG','FontSize',10,'FontWeight','bold');
drawnow;

%% GPCA - Geodesic Surface Approach

display('Geodesic surface approach')

L = 2; % Dimension of the geodesic subset
% Choose initialization
V0 = rand(L,length(Omega)+1); %random initialization
%V0(:,(f > 0)) = bsxfun(@rdivide,V0(:,nonzero_ind),sqrt(f(nonzero_ind))); % initialization with the eigenVector of the covariance matrix

[v_gpca_S, t_gpca_S,residual_S] = algo_GPCA_1D_surface(V,OmegaExt,L,V0,f);


% Representation of the 1st, 2nd components and the principal geodesic surface of
% the geodesic surface approach
figure(11)
% 1st component
subaxis(1,3,1,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
hS = zeros(size(mu,1),length(OmegaExt));
[~, I] = sort(t_gpca_S(:,1));
for i=1:size(mu,1)
    TS = OmegaExt+t_gpca_S(I(i),1)*v_gpca_S(1,:);
    hS(i,:) =pushforward_density(TS,f,OmegaExt);
    plot(OmegaExt,hS(i,:),'Color',map1(i,:))
    axis([Omega(1) Omega(end) -0.002 0.047])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
ylabel('Geodesic Surface approach','FontSize',10,'FontWeight','bold');
title('First PG','FontSize',10,'FontWeight','bold');

% 2nd component
subaxis(1,3,2,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
[~, I] = sort(t_gpca_S(:,2));
for i=1:size(mu,1)
    TS = OmegaExt+t_gpca_S(I(i),2)*v_gpca_S(2,:);
    hS(i,:) =pushforward_density(TS,f,OmegaExt);
    plot(OmegaExt,hS(i,:),'Color',map2(i,:))
    axis([Omega(1) Omega(end) -0.002 0.047])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('Second PG','FontSize',10,'FontWeight','bold');

% Principal Surface
subaxis(1,3,3,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
U=t_gpca_S*v_gpca_S;
for i=1:size(mu,1)
    TS(i,:) = OmegaExt+U(I(i),:);
    hS(i,:) =pushforward_density(TS(i,:),f,OmegaExt);
    plot(OmegaExt,hS(i,:),'Color',map3(i,:))
    axis([Omega(1) Omega(end) -0.002 0.047])
    hold on
end
pl=plot(OmegaExt,f,'-k','linewidth',3);
legend(pl,'Wasserstein barycenter');
title('Surface PG','FontSize',10,'FontWeight','bold');
drawnow;


%% Plot transport map
% Comparison between projections of the data onto iterative PG and log-PC

display('Comparison between projections of the data onto iterative PG and log-PC')
l=1;
[~, I] = sort(tV(:,l));
figure(12)
for i=1:size(mu,1)
    clf
    T_iter=OmegaExt+t_gpca_iter(I(i),l)*v_gpca_iter(l,:);
    TV=OmegaExt+Vc+tV(I(i),l)*eigV(:,l)';
    
    subaxis(1,2,1,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
    plot(OmegaExt,[T_iter ;TV]);
    legend('Map from iterative geodesic approach','Map from log-PCA approach')

    subaxis(1,2,2,'SpacingVert',0.07,'SpacingHoriz',0.07,'ML',0.15);
    bar(OmegaExt,[mu(I(i),1) mu(I(i),:)],'y');
    hold on

    h_iter=pushforward_density(T_iter,f,OmegaExt);
    hV=pushforward_density(TV,f,OmegaExt);
    
    plot(OmegaExt,h_iter,'-b','Linewidth',2)
    plot(OmegaExt,hV,'--r','Linewidth',2)
    legend('Data',['Projection onto iterative PG ',num2str(l)],['Projection onto log-PC ',num2str(l)])
    drawnow
end
title('Comparison between projections of the data onto iterative PG and log-PC')
