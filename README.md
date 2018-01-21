# 2017-GPCA-vs-LogPCA-Wasserstein

Matlab code to reproduce the results of the paper
'Geodesic PCA versus Log-PCA of histograms in the Wasserstein space', E. Cazelles, V. Seguy, J. Bigot, M. Cuturi, N. Papadakis

Arxiv: https://arxiv.org/abs/1708.08143
 
 Content :
 
 test_GPCA_vs_logPCA.m : main script to launch the computation for Gaussian data. Display the figures :
 
     1- Gaussian data    
     2- True Wasserstein barycenter of the data
     3- Data projection along principal component in Euclidean PCA     
     4- Reprensentation of the 1st and 2nd components of the Euclidean PCA    
     5- Smooth barycenter of the data     
     6- Log-maps of the data at the barycenter     
     7- Exponential map of the data at the barycenter     
     8- Data projection along principal component in log-PCA    
     9- Representation of the 1st and 2nd components of log-PCA    
     10- Representation of the 1st and 2nd components and the principal geodesic surface of the iterative geodesic approach    
     11- Representation of the 1st and 2nd components and the principal geodesic surface of the geodesic surface approach   
     12- Comparison between projections of the data onto iterative PG and log-PC

 
 algo_GPCA_1D_surface.m : Compute principal geodesics via  Geodesic surface approach
 
 algo_GPCA_1D_iter.m : Compute principal geodesics via Iterative Geodesic approach
 
 toolbox/ : various helper functions
