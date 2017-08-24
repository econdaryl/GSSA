
#Function library for the GSSA algorithm. 


from numpy import *
import math
from time import clock


# -------------------------------------------------------------------------
# Inputs:  "N" is the number of random variables; N>=1;
#          "vcv" is the variance-covariance matrix; N-by-N
#
# Outputs: "n_nodes" is the total number of integration nodes; 2*N;
#          "epsi_nodes" are the integration nodes; n_nodes-by-N;
#          "weight_nodes" are the integration weights; n_nodes-by-1
# -------------------------------------------------------------------------
def Monomials_1(N,vcv):#function [n_nodes,epsi_nodes,weight_nodes] = Monomials_1(N,vcv)
	vcv = matrix(vcv)					#
	n_nodes = 2*N #n_nodes   = 2*N;       % Total number of integration nodes
	#
	#% 1. N-dimensional integration nodes for N uncorrelated random variables with 
	#% zero mean and unit variance
	#% ---------------------------------------------------------------------------   
	z1 = matrix(zeros((n_nodes,N)))#z1 = zeros(n_nodes,N); % A supplementary matrix for integration nodes; 
	                       #% n_nodes-by-N
	                       
	for i in range(1,N+1):    #for i = 1:N      # In each node, random variable i takes value either
		z1[2*(i-1):2*i,i-1]= matrix([[1],[-1]])                         # 1 or -1, and all other variables take value 0
		#z1(2*(i-1)+1:2*i,i) = [1; -1]; 
	                        # For example, for N = 2, z1 = [1 0; -1 0; 0 1; 0 -1]
	#
	#% z = z1*sqrt(N);      % Integration nodes  
	#
	#% 2. N-dimensional integration nodes and weights for N correlated random 
	#% variables with zero mean and variance-covaraince matrix vcv 
	#% ----------------------------------------------------------------------  
	sqrt_vcv = 	linalg.cholesky(vcv).T #sqrt_vcv = chol(vcv);  % Cholesky decomposition of the variance-covariance  matrix
	R = sqrt(N)*sqrt_vcv #R = sqrt(N)*sqrt_vcv;  % Variable R; see condition (B.7) in the Supplement 
	#                       % to JMM (2011)
	#                                 
	epsi_nodes = dot(z1,R)  #epsi_nodes = z1*R;     % Integration nodes; see condition ((B.7) in the 
							#Supplement% to JMM (2011); n_nodes-by-N
	
	#% 3. Integration weights
	#%-----------------------
	weight_nodes = matrix(ones((n_nodes,1)))/double(n_nodes) #weight_nodes = ones(n_nodes,1)/n_nodes; 
	#                       % Integration weights are equal for all integration 
	#                       % nodes; n_nodes-by-1; the weights are the same for 
	#                       % the cases of correlated and uncorrelated random 
	#                       % variables
	return n_nodes,epsi_nodes,weight_nodes
	
# -------------------------------------------------------------------------
# Inputs:  "N" is the number of random variables; N>=1;
#          "vcv" is the variance-covariance matrix; N-by-N;
#
# Outputs: "n_nodes" is the total number of integration nodes; 2*N^2+1;
#          "epsi_nodes" are the integration nodes; n_nodes-by-N;
#          "weight_nodes" are the integration weights; n_nodes-by-1
# -------------------------------------------------------------------------
def Monomials_2(N,vcv): #function [n_nodes,epsi_nodes,weight_nodes] = Monomials_2(N,vcv)
	vcv = matrix(vcv)
	n_nodes = 2*pow(N,2)+1    # Total number of integration nodes

	# 1. N-dimensional integration nodes for N uncorrelated random variables with 
	# zero mean and unit variance
	# ---------------------------------------------------------------------------   
	
	# 1.1 The origin point
	# --------------------
	z0 = matrix(zeros((1,N)))       # A supplementary matrix for integration nodes: the origin point 

	# 1.2 Deviations in one dimension
	# -------------------------------
	z1 = matrix(zeros((2*N,N)))  #z1 = zeros(2*N,N);    # A supplementary matrix for integration nodes n_nodes-by-N
							
	for i in range(1,N+1): #for i = 1:N           # In each node, random variable i takes value either 1 or -1, and all other variables take value 0
		z1[2*(i-1),i-1]= 1  #CHECK THIS #z1(2*(i-1)+1:2*i,i) = [1; -1];   For example, for N = 2, z1 = [1 0; -1 0; 0 1; 0 -1]
		z1[2*i-1,i-1] = -1
	# 1.3 Deviations in two dimensions
	# --------------------------------
	z2 = matrix(zeros((2*N*(N-1),N))); #z2 = zeros(2*N*(N-1),N);  # A supplementary matrix for integration nodes; 
				   # 2N(N-1)-by-N
	
	i=0;                       # In each node, a pair of random variables (p,q)
				   # takes either values (1,1) or (1,-1) or (-1,1) or    
				   # (-1,-1), and all other variables take value 0 
	for p in range(1,N): #for p = 1:N-1                           
	    for q in range(p+1,N+1): #for q = p+1:N      
		i=i+1;               
		#z2(4*(i-1)+1:4*i,p) = [1;-1;1;-1];    
		#z2(4*(i-1)+1:4*i,q) = [1;1;-1;-1];
		
		#z2[4*(i-1):4*i,p-1] = array([1,-1,1,-1])
		#z2[4*(i-1):4*i,q-1] = array([1,1,-1,-1])
		
		z2[4*(i-1),p-1] = 1
		z2[4*(i-1)+1,p-1] = -1
		z2[4*(i-1)+2,p-1] = 1
		z2[4*(i-1)+3,p-1] = -1
		
		z2[4*(i-1),q-1] = 1
		z2[4*(i-1)+1,q-1] = 1
		z2[4*(i-1)+2,q-1] = -1
		z2[4*(i-1)+3,q-1] = -1
		
	                  # For example, for N = 2, z2 = [1 1;1 -1;-1 1;-1 1]
	
	# z = [z0;z1*sqrt(N+2);z2*sqrt((N+2)/2)];   # Integration nodes 
			       
	# 2. N-dimensional integration nodes and weights for N correlated random 
	# variables with zero mean and variance-covaraince matrix vcv 
	# ----------------------------------------------------------------------  
	sqrt_vcv = 	linalg.cholesky(vcv).T #sqrt_vcv = chol(vcv);            # Cholesky decomposition of the variance-
					 # covariance matrix
					 
	R = sqrt(double(N)+2)*sqrt_vcv #R = sqrt(N+2)*sqrt_vcv;          # Variable R; see condition (B.8) in the  
					 # Supplement to JMM (2011)
					 
	S = sqrt((double(N)+2)/2)* sqrt_vcv #S = sqrt((N+2)/2)* sqrt_vcv;     # Variable S; see condition (B.8) in the  
					 # Supplement to JMM (2011)
					 
	epsi_nodes = concatenate((z0,dot(z1,R),dot(z2,S))) #epsi_nodes = [z0;z1*R;z2*S]; 
					 # Integration nodes; see condition (B.8)   
					 # in the Supplement to JMM (2011); 
					 # n_nodes-by-N
	
	# 3. Integration weights
	#-----------------------
	weight_nodes = concatenate((2./(double(N)+2)*matrix(ones((z0.shape[0],1))),\
					(4-N)/2./pow(N+2,2)*matrix(ones((z1.shape[0],1))),\
					1./pow(N+2,2)*matrix(ones((z2.shape[0],1)))))
					#[2/(N+2)*ones(size(z0,1),1);(4-N)/2/(N+2)^2*ones(size(z1,1),1);1/(N+2)^2*ones(size(z2,1),1)];
					 # See condition in (B.8) in the Supplement  
					 # to JMM (2011); n_nodes-by-1; the weights 
					 # are the same for the cases of correlated  
					 # and uncorrelated random variables
	return n_nodes,epsi_nodes,weight_nodes

# -------------------------------------------------------------------------
# Inputs:  "T" is the simulation length; T>=1;
#          "N" is the number of countries; N>=1;
#          "a_init" is the initial condition for the productivity levels of
#          N countries; 1-by-N;
#          "rho" and "sigma" are the parameters of the model
#
# Output:  "a" are the time series of the productivity levels of N countries; 
#          T-by-N
# -------------------------------------------------------------------------
def Productivity(T,N,a_init,sigma,rho): #function a = Productivity(T,N,a_init,sigma,rho)

	EPSI = matrix(random.randn(T,1)) #EPSI = randn(T,1);  % A random draw of common-for-all-countries productivity 
					   #shocks for T periods; T-by-1 
						
	epsi = matrix(random.randn(T,N))  #epsi = randn(T,N);  % A random draw of country-specific productivity shocks 
						# for T periods and N countries; T-by-N

	epsi = (epsi+dot(EPSI,ones((1,N))))*float(sigma) #epsi = (epsi+EPSI*ones(1,N))*sigma; 
						# Compute the error terms in the process for productivity 
						# level using condition (4) in JMM (2011); T-by-N
	a = matrix(zeros((T,N)));
	a[0,0:N] = a_init	#a(1,1:N) = a_init;   %Initial condition for the productivity levels; 1-by-N

	
	for t in range(1,T): #for t = 1:T-1; 
		a[t,:] = multiply(power(a[t-1,:],rho),exp(epsi[t,:])) #a(t+1,:) = a(t,:).^rho.*exp(epsi(t+1,:)); 
						# Compute the next-period productivity levels using 
						# condition (4) in JMM (2011); 1-by-N 
	return a
	
# -------------------------------------------------------------------------
# Inputs:  "z" is the data points on which the polynomial basis functions  
#               must be constructed; n_rows-by-dimen; 
#          "D" is the degree of the polynomial whose basis functions must 
#               be constructed; (can be 1,2,3,4 or 5)
#
# Output:  "basis_fs" is the matrix of basis functions of a complete 
#               polynomial of the given degree 
# -------------------------------------------------------------------------
def Ord_Polynomial_N(z,D): #function basis_fs = Ord_Polynomial_N(z,D)
# A polynomial is given by the sum of polynomial basis functions, phi(i),
# multiplied by the coefficients; see condition (13) in JMM (2011). By 
# convention, the first basis function is one. 
	n_rows = z.shape[0] #[n_rows,dimen] = size(z); % Compute the number of rows, n_rows, and the  
	dimen = z.shape[1]        # number of variables (columns), dimen, in the    
							  # data z on which the polynomial basis functions   
							  # must be constructed
    # 1. The matrix of the basis functions of the first-degree polynomial 
    # (the default option)
    # -------------------------------------------------------------------
	basis_fs = concatenate((ones((n_rows,1)),z),1) 
	#basis_fs = [ones(n_rows,1) z];  % The matrix includes a column of ones
                                    # (the first basis function is one for
                                    # n_rows points) and linear basis
                                    # functions
	i = dimen+1 #i = dimen+1; % Index number of a polynomial basis function; the first  
						      # basis function (equal to one) and linear basis functions 
							  # are indexed from 1 to dimen+1, and subsequent polynomial 
							  # basis functions will be indexed from dimen+2 and on
    
    # 2. The matrix of the basis functions of the second-degree polynomial 
    # --------------------------------------------------------------------   
	if D == 2: 
# Version one (not vectorized): 
		for j1 in range(1,dimen+1): #for j1 = 1:dimen   
			for j2 in range(j1,dimen+1): #for j2 = j1:dimen
				#i = i +1 #i = i+1;
				#basis_fs = concatenate((basis_fs,multiply(z[:,(j1-1):j1],z[:,(j2-1):j2])),1) 
				basis_fs = concatenate((basis_fs,multiply(z[:,j1-1],z[:,j2-1])),1) 
				#basis_fs(:,i) = z(:,j1).*z(:,j2);
           
# % Version 2 (vectorized): Note that this version works only for a second-degree 
# % polynomial in which all state variables take non-zero values
#         for r = 1:n_rows
#             basis_fs(r,2+dimen:1+dimen+dimen*(dimen+1)/2) = [nonzeros(tril(z(r,:)'*z(r,:)))']; 
#             % Compute linear and quadratic polynomial basis functions for 
#             % each row r; "tril" extracts a lower triangular part of z'z 
#             % (note that matrix z'z is symmetric so that an upper triangular 
#             % part is redundant); "nonzeros" forms a column vector by 
#             % stacking the columns of the original matrix one after another 
#             % and by eliminating zero terms
#         end
 
    # 3. The matrix of the basis functions of the third-degree polynomial 
    # -------------------------------------------------------------------    
	elif D==3: #elseif D == 3                
		for j1 in range(1,dimen+1): #for j1 = 1:dimen   
			for j2 in range(j1,dimen+1): #for j2 = j1:dimen
				#i = i +1 #i = i+1;
				basis_fs = concatenate((basis_fs,multiply(z[:,(j1-1):j1],z[:,(j2-1):j2])),1)
				#basis_fs(:,i) = z(:,j1).*z(:,j2);
				for j3 in range(j2,dimen+1):#for j3 = j2:dimen
					#i = i +1 #i = i+1;
					basis_fs = concatenate((basis_fs,multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);

					
    # 4. The matrix of the basis functions of the fourth-degree polynomial 
    # -------------------------------------------------------------------    
	elif D==4: #elseif D == 3                
		for j1 in range(1,dimen+1): #for j1 = 1:dimen   
			for j2 in range(j1,dimen+1): #for j2 = j1:dimen
				i = i +1 #i = i+1;
				basis_fs = concatenate((basis_fs,multiply(z[:,(j1-1):j1],z[:,(j2-1):j2])),1)#basis_fs(:,i) = z(:,j1).*z(:,j2);
				for j3 in range(j2,dimen+1):#for j3 = j2:dimen
					i = i +1 #i = i+1;
					basis_fs = concatenate((basis_fs,multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);
					for j4 in range(j3,dimen+1): #for j4 = j3:dimen
						i = i +1 #i = i+1;
						basis_fs = concatenate((basis_fs,multiply(multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3]),z[:,(j4-1):j4])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4);

    # 5. The matrix of the basis functions of the fifth-degree polynomial 
    # ------------------------------------------------------------------- 
	elif D == 5: #elseif D == 5                          
		for j1 in range(1,dimen+1): #for j1 = 1:dimen   
			for j2 in range(j1,dimen+1): #for j2 = j1:dimen
				i = i +1 #i = i+1;
				basis_fs = concatenate((basis_fs,multiply(z[:,(j1-1):j1],z[:,(j2-1):j2])),1)#basis_fs(:,i) = z(:,j1).*z(:,j2);
				for j3 in range(j2,dimen+1):#for j3 = j2:dimen
					i = i +1 #i = i+1;
					basis_fs = concatenate((basis_fs,multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);
					for j4 in range(j3,dimen+1): #for j4 = j3:dimen
						i = i +1 #i = i+1;
						basis_fs = concatenate((basis_fs,multiply(multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3]),z[:,(j4-1):j4])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4);
						for j5 in range(j4,dimen+1): #for j5 = j4:dimen
							i = i +1 #i = i+1;
							basis_fs = concatenate((basis_fs,multiply(multiply(multiply(multiply(z[:,(j1-1):j1],z[:,(j2-1):j2]),z[:,(j3-1):j3]),z[:,(j4-1):j4]),z[:,(j5-1):j5])),1) #basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4).*z(:,j5);
                        
	return basis_fs
	
from numpy import *

# -------------------------------------------------------------------------
# Inputs:  "z" is the data points on which the polynomial basis functions  
#          must be constructed; T-by-2; 
#          "D"  is the degree of the polynomial whose basis functions must 
#          be constructed; (can be 1,2,3,4 or 5)
#          "PF" is the polynomial family chosen; 0=Ordinary, 1=Hermite;
#          "zb" is Matrix of means and standard deviations of state
#          variables; it is used to normalize these variables in the 
#          Hermite polynomial; 2-by-2            
# 
# Output:  "basis_fs" is the matrix of basis functions of a complete 
#          polynomial of the given degree 
# -------------------------------------------------------------------------
def  Ord_Herm_Pol_1(z,D,PF,zb): #function basis_fs = Ord_Herm_Pol_1(z,D,PF,zb)
# A polynomial is given by the sum of polynomial basis functions, phi(i),
# multiplied by the coefficients; see condition (13) in JMM (2011). By 
# convention, the first basis function is one. 
	z = matrix(z)
	n_rows = z.shape[0] #n_rows = size(z,1);   % Infer the number of rows in the data z on which the 
                      # polynomial basis functions must be constructed
	
	if PF == 1: #if PF == 1;      % If the polynomial family chosen is Hermite, ...
		zc1 = (z[:,0]-zb[0,0])/double(zb[1,0]) 
		#zc1 = (z(:,1)-zb(1,1))/zb(2,1); 
				# Normalize the variable z(:,1); n_rows-by-1
		zc2 = (z[:,1]-zb[0,1])/double(zb[1,1]) 
		#zc2 = (z(:,2)-zb(1,2))/zb(2,2); 
				# Normalize the variable z(:,2); n_rows-by-1
		p1 = zc1         #p1 = zc1; p2 = zc1.^2-1; p3 = zc1.^3-3*zc1; p4 = zc1.^4-6*zc1.^2+3; p5 = zc1.^5-10*zc1.^3+15*zc1;
		p2 = power(zc1,2)-1
		p3 = power(zc1,3)-3*zc1
		p4 = power(zc1,4)-6*power(zc1,2)+3
		p5 = power(zc1,5)-10*power(zc1,3)+15*zc1
		# p1,...,p5 are the vectors obtained by evaluating the Hermite 
				# polynomial basis functions from the second to sixth 
				# (excluding the first basis function equal to one) in all 
				# values of zc1; each vector is n_rows-by-1
		#q1 = zc2; q2 = zc2.^2-1; q3 = zc2.^3-3*zc2; q4 = zc2.^4-6*zc2.^2+3; q5 = zc2.^5-10*zc2.^3+15*zc2;
		q1 = zc2
		q2 = power(zc2,2)-1
		q3 = power(zc2,3)-3*zc2
		q4 = power(zc2,4)-6*power(zc2,2)+3
		q5 = power(zc2,5)-10*power(zc2,3)+15*zc2
		# q1,...,q5 are the vectors obtained by evaluating the Hermite 
				# polynomial basis functions from the second to sixth 
				# (excluding the first basis function equal to one) in all 
				# values of zc2; each vector is n_rows-by-1
	else:             # If the polynomial family chosen is ordinary, ...
		zc1 = z[:,0] #zc1 = z(:,1); # No normalization for z(:,1); n_rows-by-1 
		zc2 = z[:,1] #zc2 = z(:,2); # No normalization for z(:,2); n_rows-by-1
		#p1 = zc1; p2 = zc1.^2; p3 = zc1.^3; p4 = zc1.^4; p5 = zc1.^5;
		p1 = zc1
		p2 = power(zc1,2)
		p3 = power(zc1,3)
		p4 = power(zc1,4)
		p5 = power(zc1,5)
				# p1,...,p5 are the vectors obtained by evaluating the ordinary 
				# polynomial basis functions from the second to sixth 
				# (excluding the first basis function equal to one) in all 
				# values of zc1; each vector is n_rows-by-1
				
		#q1 = zc2; q2 = zc2.^2; q3 = zc2.^3; q4 = zc2.^4; q5 = zc2.^5;
		q1 = zc2
		q2 = power(zc2,2)
		q3 = power(zc2,3)
		q4 = power(zc2,4)
		q5 = power(zc2,5)
		
		# q1,...,q5 are the vectors obtained by evaluating the ordinary 
                 # polynomial basis functions from the second to sixth 
                 # (excluding the first basis function equal to one) in all 
                 # values of zc2; each vector is n_rows-by-1



# Construct the matrix of the basis functions
#--------------------------------------------
	if D == 1: #if D == 1;        
		basis_fs = concatenate((matrix(ones((n_rows,1))),p1,q1),1) 
		#basis_fs = [ones(n_rows,1) p1 q1]; 
			   # The matrix of basis functions of the first-degree polynomial
	elif D == 2: #elseif D == 2;
		basis_fs = concatenate((matrix(ones((n_rows,1))), p1, q1, p2, multiply(p1,q1), q2),1) 
		#basis_fs = [ones(n_rows,1) p1 q1 p2 p1.*q1 q2];
			   # The matrix of basis functions of the second-degree polynomial
	elif D == 3: #elseif D == 3;
		basis_fs = concatenate((matrix(ones((n_rows,1))), p1, q1, p2, multiply(p1,q1), q2, p3, multiply(p2,q1), multiply(p1,q2), q3),1) #basis_fs = [ones(n_rows,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3];
			   # The matrix of basis functions of the third-degree polynomial               
	elif D == 4: #elseif D == 4;
		basis_fs = concatenate((matrix(ones((n_rows,1))), p1, q1, p2, multiply(p1,q1), q2, p3, multiply(p2,q1), multiply(p1,q2), q3, p4, multiply(p3,q1), multiply(p2,q2), multiply(p1,q3), q4),1)
		#basis_fs = [ones(n_rows,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3 p4 p3.*q1 p2.*q2 p1.*q3 q4];
			   # The matrix of basis functions of the fourth-degree polynomial
	elif D == 5: #elseif D == 5;
		basis_fs = concatenate((matrix(ones((n_rows,1))), p1, q1, p2, multiply(p1,q1), q2, p3, multiply(p2,q1), multiply(p1,q2), q3, p4, multiply(p3,q1), multiply(p2,q2), multiply(p1,q3), q4, p5, multiply(p4,q1), multiply(p3,q2), multiply(p2,q3), multiply(p1,q4), q5),1)#basis_fs = [ones(n_rows,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3 p4 p3.*q1 p2.*q2 p1.*q3 q4 p5 p4.*q1 p3.*q2 p2.*q3 p1.*q4 q5];
			   # The matrix of basis functions of the fifth-degree polynomial
	return basis_fs
	
	
# -------------------------------------------------------------------------
# Inputs:  "Qn" is the number of nodes in each dimension; 1<=Qn<=10;
#          "N" is the number of random variables; N=1,2,...;
#          "vcv" is the variance-covariance matrix; N-by-N
#
# Outputs: "n_nodes" is the total number of integration nodes; Qn^N;
#          "epsi_nodes" are the integration nodes; n_nodes-by-N;
#          "weight_nodes" are the integration weights; n_nodes-by-1
# -------------------------------------------------------------------------
def GH_Quadrature(Qn,N,vcv):
	vcv = matrix(vcv)
	# 1. One-dimensional integration nodes and weights (given with 16-digit 
	# accuracy) under Gauss-Hermite quadrature for a normally distributed random 
	# variable with zero mean and unit variance
	# -------------------------------------------------------------------------
	if Qn == 1:                 # Number of nodes in each dimension; Qn <=10
		eps = matrix([[0]])             # Set of integration nodes
		weight = matrix([[sqrt(pi)]])   # Set of integration weights      
	elif Qn == 2:            
		eps = matrix([[0.7071067811865475], [-0.7071067811865475]]) 
		weight = matrix([[0.8862269254527580],[0.8862269254527580]]);
	elif Qn == 3:
		eps = matrix([[1.224744871391589], [0], [-1.224744871391589]]);
		weight = matrix([[0.2954089751509193],[1.181635900603677],[0.2954089751509193]]);
	elif Qn == 4:
		eps = matrix([[1.650680123885785],[0.5246476232752903],[-0.5246476232752903],[-1.650680123885785]]);
		weight = matrix([[0.08131283544724518],[0.8049140900055128],[0.8049140900055128],[0.08131283544724518]]);
	elif Qn == 5:
		eps = matrix([[2.020182870456086],[0.9585724646138185],[0],[-0.9585724646138185],[-2.020182870456086]]);
		weight = matrix([[0.01995324205904591],[0.3936193231522412],[0.9453087204829419],[0.3936193231522412],[0.01995324205904591]]);
	elif Qn == 6:
		eps = matrix([[2.350604973674492],[1.335849074013697],[0.4360774119276165],[-0.4360774119276165],[-1.335849074013697],[-2.350604973674492]]);
		weight = matrix([[0.004530009905508846],[0.1570673203228566],[0.7246295952243925],[0.7246295952243925],[0.1570673203228566],[0.004530009905508846]]);
	elif Qn == 7:
		eps = matrix([[2.651961356835233],[1.673551628767471],[0.8162878828589647],[0],[-0.8162878828589647],[-1.673551628767471],[-2.651961356835233]]);
		weight = matrix([[0.0009717812450995192], [0.05451558281912703],[0.4256072526101278],[0.8102646175568073],[0.4256072526101278],[0.05451558281912703],[0.0009717812450995192]]); 
	elif Qn == 8:
		eps = matrix([[2.930637420257244],[1.981656756695843],[1.157193712446780],[0.3811869902073221],[-0.3811869902073221],[-1.157193712446780],[-1.981656756695843],[-2.930637420257244]]);
		weight = matrix([[0.0001996040722113676],[0.01707798300741348],[0.2078023258148919],[0.6611470125582413],[0.6611470125582413],[0.2078023258148919],[0.01707798300741348],[0.0001996040722113676]]); 
	elif Qn == 9:
		eps = matrix([[3.190993201781528],[2.266580584531843],[1.468553289216668],[0.7235510187528376],[0],[-0.7235510187528376],[-1.468553289216668],[-2.266580584531843],[-3.190993201781528]])
		weight = matrix([[0.00003960697726326438],[0.004943624275536947],[0.08847452739437657],[0.4326515590025558],[0.7202352156060510],[0.4326515590025558],[0.08847452739437657],[0.004943624275536947],[0.00003960697726326438]]);    
	else:
		Qn = 10 # The default option
		eps = matrix([[3.436159118837738],[2.532731674232790],[1.756683649299882],[1.036610829789514],[0.3429013272237046],[-0.3429013272237046],[-1.036610829789514],[-1.756683649299882],[-2.532731674232790],[-3.436159118837738]]);
		weight = matrix([[7.640432855232621e-06],[0.001343645746781233],[0.03387439445548106],[0.2401386110823147],[0.6108626337353258],[0.6108626337353258],[0.2401386110823147],[0.03387439445548106],[0.001343645746781233],[7.640432855232621e-06]]);

	
	# 2. N-dimensional integration nodes and weights for N uncorrelated normally 
	# distributed random variables with zero mean and unit variance
	# ------------------------------------------------------------------------                        
	n_nodes = power(Qn,N);  #Qn^n      % Total number of integration nodes (in N dimensions)
	
	z1 = matrix(zeros((n_nodes,N)))#z1 = zeros(n_nodes,N); % A supplementary matrix for integration nodes; 
						# n_nodes-by-N 
	w1 = matrix(ones((n_nodes,1)))#w1 = ones(n_nodes,1);  % A supplementary matrix for integration weights; 
						# n_nodes-by-1
	
	for i in range(1,N+1): #for i = 1:N            
		z1i = matrix([[1]]) #z1i = [];           # A column for variable i to be filled in with nodes 
		w1i = matrix([[1]]) #w1i = [];           # A column for variable i to be filled in with weights 
		for j in range(1,Qn**(N-i)+1): #for j = 1:Qn^(N-i)
			for u in range(1,Qn+1): #for u=1:Qn
				z1i = concatenate((z1i,ones((Qn**(i-1),1))*eps[u-1])) #z1i = [z1i;ones(Qn^(i-1),1)*eps(u)];
				w1i = concatenate((w1i,ones((Qn**(i-1),1))*weight[u-1])) #w1i = [w1i;ones(Qn^(i-1),1)*weight(u)];
				
		z1[:,(i-1):i] = z1i[1:z1i.size] #z1(:,i) = z1i;      % z1 has its i-th column equal to z1i 
		w1 = multiply(w1,w1i[1:w1i.size]) #w1 = w1.*w1i;       % w1 is a product of weights w1i 

	
	z = sqrt(2)*z1 #z = sqrt(2).*z1;       % Integration nodes; n_nodes-by-N; for example, 
						# for N = 2 and Qn=2, z = [1 1; -1 1; 1 -1; -1 -1]
	
	w = w1/sqrt(math.pi)**N #w = w1/sqrt(pi)^N;     % Integration weights; see condition (B.6) in the 
						# Supplement to JMM (2011); n_nodes-by-1
	
	# 3. N-dimensional integration nodes and weights for N correlated normally 
	# distributed random variables with zero mean and variance-covariance matrix, 
	# vcv 
	# -----------------------------------------------------------------------                      
	sqrt_vcv = linalg.cholesky(vcv).T #sqrt_vcv = chol(vcv);            % Cholesky decomposition of the variance-
									# covariance matrix
									
	epsi_nodes = dot(z,sqrt_vcv) #epsi_nodes = z*sqrt_vcv;         % Integration nodes; see condition (B.6)  
									# in the Supplement to JMM (2011); 
									# n_nodes-by-N                                
	
	weight_nodes = w #weight_nodes = w;                # Integration weights are the same for the 
									# cases of correlated and uncorrelated 
									# random variables 
	return n_nodes,epsi_nodes,weight_nodes
	

# This version: July 14, 2011. First version: August 27, 2009.
# -------------------------------------------------------------------------
# Inputs:    "k" and "a" are, respectively, current-period capital and 
#            productivity levels, in the given set of points on which the 
#            accuracy is tested; 
#            "bk" are the coefficients of the capital policy function;
#            "IM" is the integration method for evaluating accuracy, 
#            IM=1,2,..,10=Gauss-Hermite quadrature rules with 1,2,...,10 
#            nodes in one dimension, respectively;
#            "PF" is the polynomial family, 0=Ordinary, 1=Hermite;
#            "zb" is a matrix of means and standard deviations of the state
#            variables, k and a; it is used to normalize these variables in 
#            the Hermite polynomial;            
#            "sigma", "rho", "beta", "gam", "alpha" and "delta" are the 
#            parameters of the model;
#            "discard" is the number of data points to discard 
#
# Outputs:   "Errors_mean" and "Errors_max" are, respectively, the mean and
#            maximum absolute Euler equation errors (in log10)

# -------------------------------------------------------------------------
# Inputs:    "k" and "a" are, respectively, current-period capital and 
#            productivity levels, in the given set of points on which the 
#            accuracy is tested; 
#            "bk" are the coefficients of the capital policy function;
#            "IM" is the integration method for evaluating accuracy, 
#            IM=1,2,..,10=Gauss-Hermite quadrature rules with 1,2,...,10 
#            nodes in one dimension, respectively;
#            "PF" is the polynomial family, 0=Ordinary, 1=Hermite;
#            "zb" is a matrix of means and standard deviations of the state
#            variables, k and a; it is used to normalize these variables in 
#            the Hermite polynomial;            
#            "sigma", "rho", "beta", "gam", "alpha" and "delta" are the 
#            parameters of the model;
#            "discard" is the number of data points to discard 
#
# Outputs:   "Errors_mean" and "Errors_max" are, respectively, the mean and
#            maximum absolute Euler equation errors (in log10)
# -------------------------------------------------------------------------
def Accuracy_Test_1(sigma,rho,beta,gam,alpha,delta,k,a,bk,D,IM,PF,zb,discard): #function [Errors_mean Errors_max time_test]  = Accuracy_Test_1(sigma,rho,beta,gam,alpha,delta,k,a,bk,D,IM,PF,zb,discard)
	tic = clock() #tic              # Start counting time needed to run the test
	[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(IM,1,sigma**2) #[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(IM,1,sigma^2);
					# n_nodes is the number of integration nodes, epsi_nodes  
					# are integration nodes, and weight_nodes are integration 
					# weights for Gauss-Hermite quadrature integration rule
					# with IM nodes; for a unidimensional integral, n_nodes  
					# coincides with IM

	Errors = matrix(zeros((a.shape[0],1)))
	for p in range(1,a.shape[0]+1): #for p = 1:size(a,1):     # For each point on which the accuracy is
							# evaluated, ... 
		
		# Variables in point p 
		# ------------------------        
		k0 = k[p-1,0] #k0 = k(p,1);         % Capital of period t
		a0 = a[p-1,0] #a0 = a(p,1);         % Productivity level of period t 
		
		# Capital and consumption choices at t
		# ------------------------------------  
		k1 =  dot(Ord_Herm_Pol_1(matrix([[k0,a0]]),D,PF,zb),bk) 
		#k1(1,1) =  Ord_Herm_Pol_1([k0(1,1) a0(1,1)],D,PF,zb)*bk;
		# Compute capital of period t+1 (chosen at t) using the corresponding 
		# capital policy function; 1-by-1
		c0 = k0**alpha*a0 - k1 + (1-delta)*k0 
		#c0(1,1) = k0(1,1)^alpha*a0(1,1) - k1(1,1) + (1-delta)*k0(1,1);
		# Consumption of period t
		
		# Capital and consumption choices at t+1
		#---------------------------------------
		a1 = multiply(power(a0,rho),exp(epsi_nodes)) 
		#a1 = a0.^rho.*exp(epsi_nodes); 
		# Productivity levels of period t+1; n_nodes-by-1 
		
		k1_dupl = dot(matrix(ones((n_nodes,1))),k1) #k1_dupl = ones(n_nodes,1)*k1; 
		# Duplicate k1 n_nodes times to create a matrix with n_nodes identical
		# rows; n_nodes-by-1 
		X1 = Ord_Herm_Pol_1(concatenate((k1_dupl,a1[:,0]),1),D,PF,zb) 
		
		#X1 = Ord_Herm_Pol_1([k1_dupl a1(:,1)],D,PF,zb);
		# Form a complete polynomial of degree D (at t+1) in the given point 
		k2 =  dot(X1,bk) #k2(:,1) =  X1*bk;
		# Compute capital of period t+2 (chosen at t+1) using the fifth-
		# degree capital policy function; n_nodes-by-1 

		c1 =  multiply(power(k1,alpha),a1[:,0]) - k2[:,0] + (1-delta)*k1[0,0]
	
		#c1(:,1) =  k1(1,1)^alpha*a1(:,1) - k2(:,1) + (1-delta)*k1(1,1);
		# Consumption of period t+1; n_nodes-by-1
	
		# Approximation errors in point p
		#--------------------------------  
		term1 = beta*power(c1[0:n_nodes,0],-gam)/power(c0[0,0],-gam)
		term2 = 1-delta+alpha*multiply(a1[0:n_nodes,0],power(k1[0,0],(alpha-1)))
		Errors[p-1,0] = weight_nodes.T*multiply(term1,term2)-1
		#Errors(p,1) = weight_nodes'*(beta*c1(1:n_nodes,1).^(-gam)./c0(1,1).^(-gam).*(1-delta+alpha*a1(1:n_nodes,1).*k1(1,1).^(alpha-1)))-1;
		# A unit-free Euler-equation approximation error
	#end
	
	
	Errors_mean = log10(mean(abs(Errors[discard:,:]))) #Errors_mean = log10(mean(mean(abs(Errors(1+discard:end,:)))));
	# Mean absolute Euler equation errors (in log10)
	
	Errors_max = log10(abs(Errors[discard:,:]).max()) #Errors_max = log10(max(max(abs(Errors(1+discard:end,:)))));    
	# Maximum absolute Euler equation errors (in log10)
	
	
	time_test = clock()-tic   # Time needed to run the test  
	return Errors_mean, Errors_max, time_test
	

	
# This version: July 14, 2011. First version: August 27, 2009.
# -------------------------------------------------------------------------
# Inputs:    "k" and "a" are, respectively, current-period capital and 
#            productivity levels, in the given set of points on which the 
#            accuracy is tested; 
#            "bk" are the coefficients of the capital policy functions of N 
#            countries;
#            "IM" is the integration method for evaluating accuracy, 
#            IM=1,2,..,10=Gauss-Hermite quadrature rules with 1,2,...,10 
#            nodes in each dimension, respectively, 
#            IM=11=Monomial rule with 2N nodes,
#            IM=12=Monomial rule with 2N^2+1 nodes;
#            "alpha", "gam", "phi", "beta", "A", "tau", "rho" and "vcv"
#            are the parameters of the model;
#            "discard" is the number of data points to discard 
#
# Outputs:   "Errors_mean" and "Errors_max" are, respectively, the mean and
#            maximum approximation errors across all optimality conditions;
#            "Errors_max_EE", "Errors_max_MUC", "Errors_max_MUL", and 
#            "Errors_max_RC" are the maximum approximation errors disaggre- 
#            gated by optimality conditions 
def Accuracy_Test_N(k,a,bk,D,IM,alpha,gam,delta,beta,A,tau,rho,vcv,discard):
#function [Errors_mean Errors_max time_test] = Accuracy_Test_N(k,a,bk,D,IM,alpha,gam,delta,beta,A,tau,rho,vcv,discard)

	tic = clock();  #tic                 # Start counting time needed to run the test        

	P,N = a.shape #[P,N] = size(a);      # Infer the number of points P on which the accuracy 
						# is evaluated and the number of countries N
						
	
	# 2. Integration method for evaluating accuracy 
	# ---------------------------------------------
	if (IM>=1)and(IM<=10): #if (IM>=1)&&(IM<=10)
		[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(IM,N,vcv)
								# Compute the number of integration nodes, 
								# n_nodes, integration nodes, epsi_nodes, and 
								# integration weights, weight_nodes, for Gauss-
								# Hermite quadrature integration rule with IM 
								# nodes in each dimension
	elif IM == 11: #elseif IM == 11
		[n_nodes,epsi_nodes,weight_nodes] = Monomials_1(N,vcv)
								# Monomial integration rule with 2N nodes
	elif IM == 12: #elseif IM == 12
		[n_nodes,epsi_nodes,weight_nodes] = Monomials_2(N,vcv);
								# Monomial integration rule with 2N^2+1 nodes  
	
	# 3. Polynomial bases for the test
	#---------------------------------
	X = Ord_Polynomial_N(bmat('k,a'),D) # Form a complete ordinary polynomial of degree
								# D on the given set of points 
	
	# 4. Given the solution for capital, compute consumption on the given set 
	# of points  
	#------------------------------------------------------------------------
	Errors = matrix(zeros((P,4*N+1)))
	for p in range(1,P+1): #for p = 1:P;                 % For each given point, ...     
		
		#print p       #p              # Display the point (with the purpose of  
								# monitoring the progress) 
										
		# 4.1 Variables in point p 
		# ------------------------        
		k0 = k[p-1,0:(N)]#k0 = k(p,1:N);       % N capital stocks of period t
		a0 = a[p-1,0:(N)]#a0 = a(p,1:N);       % N productivity levels of period t
		X0 = X[p-1,:]#X0 = X(p,:);         % Complete (second-degree) polynomial 
							# bases at t
									
		# 4.2 Capital and consumption choices at t
		# ----------------------------------------
		k1 = dot(X0,bk) #k1 = X0*bk; 
		# Compute a row-vector of capital of period t+1 (chosen at t) using
		# the corresponding capital policy functions; 1-by-N
		
		C0 = (A*multiply(power(k0,alpha),a0)-k1+k0*(1-delta))*ones((N,1))#C0 = (A*k0.^alpha.*a0 - k1+k0*(1-delta))*ones(N,1);
		# C is computed by summing up individual consumption, which in turn, is 
		# found from the individual budget constraints; 1-by-1
	
		c0 = dot(C0,matrix(ones((1,N))))/double(N)#c0 = C0*ones(1,N)/N;  # Individual consumption is the same for all
								# countries; 1-by-N 
	
		# 4.3 Capital and consumption choices at t+1
		#-------------------------------------------
		a1 = multiply(power((matrix(ones((n_nodes,1)))*a0),rho),exp(epsi_nodes)) #a1 = (ones(n_nodes,1)*a0).^rho.*exp(epsi_nodes);    
		# Compute the next-period productivity levels in each integration node
		# using condition (?) in the online appendix; n_nodes-by-N
	
		k1_dupl = dot(ones((n_nodes,1)),k1) #k1_dupl = ones(n_nodes,1)*k1; 
		# Duplicate k1 n_nodes times to create a matrix with n_nodes identical
		# rows; n_nodes-by-N 
	
		X1 = Ord_Polynomial_N(bmat('k1_dupl, a1'),D) #X1 = Ord_Polynomial_N([k1_dupl a1],D);
		# Form a complete polynomial of degree D (at t+1) in the given point 
		
		k2 = dot(X1,bk) #k2 = X1*bk; 
		# Compute capital of period t+2 (chosen at t+1) using the second-
		# degree capital policy functions; n_nodes-by-N 
	
		C1 = (A*multiply(power(k1_dupl,alpha),a1) - k2+k1_dupl*(1-delta))*ones((N,1)) #C1 = (A*k1_dupl.^alpha.*a1 - k2+k1_dupl*(1-delta))*ones(N,1);
		# Aggregate consumption is computed by summing up individual consumption, 
		# which in turn, is found from the individual budget constraints; 
		# n_nodes-by-1
	
		c1 = dot(C1,matrix(ones((1,N))))/double(N) #c1 = C1*ones(1,N)/N;    # Individual consumption is the same for 
									# all countries; n_nodes-by-N
	
	# 5. Approximation errors in point p
	#-----------------------------------
			
			# 5.1 Lagrange multiplier associated with the aggregate resource
			# constraint
			#---------------------------------------------------------------
		MUC0j = matrix(zeros((1,N)))
		for j in range(1,N+1): #for j = 1:N
			MUC0j[0,j-1] = tau*power(c0[0,j-1],(-gam)) #MUC0j(1,j) = tau*c0(1,j).^(-gam); 
			# Compute a country's marginal utility of consumption multiplied 
			# by its welfare weight
		#end
		lambda0 = mean(MUC0j,axis=1) #lambda0 = mean(MUC0j,2);
		# An optimality condition w.r.t. consumption of period t equates 
		# the Lagrange multiplier of the aggregate resource constraint of 
		# period t and each country's marginal utility of consumption 
		# multiplied by its welfare weight; to infer the Lagrange multiplier,  
		# we average across N countries; 1-by-1
		MUC1j = matrix(zeros((n_nodes,N)))
		for j in range(1,N+1): #for j = 1:N
			MUC1j[0:n_nodes,j-1] = tau*power(c1[0:n_nodes,j-1],(-gam)) #MUC1j(1:n_nodes,j) = tau*c1(1:n_nodes,j).^(-gam);
			# Compute a country's marginal utility of consumption multiplied 
			# by its welfare weight
		#end
		lambda1 = mean(MUC1j,axis=1)#lambda1 = mean(MUC1j,2);
		# Similarly, the Lagrange multiplier of the aggregate resource 
		# constraint of period t+1 is equal to a country's marginal utility 
		# of consumption multiplied by its welfare weight; to infer the 
		# Lagrange multiplier, we average across N countries; 1-by-n_nodes
		
		# 5.2 Unit-free Euler-equation errors
		#------------------------------------
		
		for j in range(1,N+1): #for j = 1:N
			temp1 = beta*lambda1/lambda0
			temp2 = 1-delta+alpha*A*power(k1[0,j-1],(alpha-1))*a1[0:n_nodes,j-1]
			Errors[p-1,j-1] = 1-weight_nodes.T*multiply(temp1,temp2)
			#Errors(p,j) = 1-weight_nodes'*(beta*lambda1/lambda0.*(1-delta+alpha*A*k1(1,j)^(alpha-1)*a1(1:n_nodes,j)));
		# A unit-free Euler-equation approximation error of country j
		#end
		
		# 5.2 Unit-free errors in the optimality conditions w.r.t. consumption
		#---------------------------------------------------------------------
		for j in range(1,N+1): #for j = 1:N;
			Errors[p-1,N+j-1] = 1-lambda0/(tau*c0[0,j-1]**(-gam)) #Errors(p,N+j) = 1-lambda0./(tau*c0(1,j)^(-gam)); 
		# A unit-free approximation error in the optimality condition w.r.t. 
		# consumption of country j (this condition equates marginal utility 
		# of consumption, multiplied by the welfare weight, and the 
		# Lagrange multiplier of the aggregate resource constraint)
		# end
		
		# 5.3 Unit-free errors in the optimality conditions w.r.t. labor 
		#---------------------------------------------------------------
		Errors[p-1,(2*N):(3*N)] = matrix(zeros((1,N))) #Errors(p,2*N+1:3*N) = zeros(N,1);
		# These errors  are zero by construction 
		
		# 5.4 Unit-free approximation error in the aggregate resource constraint
		#-----------------------------------------------------------------------
		temp1 = (c0[0,0:N] + k1[0,0:N]-k0[0,0:N]*(1-delta))*ones((N,1))
		temp2 = (A*multiply(power(k0[0,0:N],alpha),a0[0,0:N]))*ones((N,1))
		Errors[p-1,3*N] = 1-temp1/temp2
		#Errors(p,3*N+1) = 1-(c0(1,1:N) + k1(1,1:N)-k0(1,1:N)*(1-delta))*ones(N,1)/((A*k0(1,1:N).^alpha.*a0(1,1:N))*ones(N,1));
		# This error is a unit-free expression of the resource constraint  
		# (?) in the online appendix
		
		# 5.5 Approximation errors in the capital-accumulation equation
		#--------------------------------------------------------------
		Errors[p-1,(3*N+1):(4*N+1)] = matrix(zeros((1,N))); #Errors(p,3*N+2:4*N+1) = zeros(N,1);
		# These errors are always zero by construction
		
		# For this model, GSSA produces zero errors (with machine accuracy)
		# in all the optimality conditions except of the Euler equation. 
		# Here, the errors in all the optimality conditions are introduced 
		# to make our accuracy measures comparable to those in the February 
		# 2011 special issue of the Journal of Economic Dynamics and Control 
	#end
	
	# 6. Mean and maximum approximation errors computed after discarding the  
	# first "discard" observations
	#-----------------------------------------------------------------------
	
	# 6.1 Approximation errors across all the optimality conditions
	#--------------------------------------------------------------
	Errors_mean = log10(mean(mean(abs(Errors[discard:,:]),axis=0),axis=1)) 
	#Errors_mean = log10(mean(mean(abs(Errors(1+discard:end,:))))); 
	# Average absolute approximation error 
	
	Errors_max = log10(abs(Errors[discard:,:]).max()) 
	#Errors_max = log10(max(max(abs(Errors(1+discard:end,:)))));    
	# Maximum absolute approximation error
	
	# 6.2 Maximum approximation errors disaggregated by the optimality 
	# conditions
	#-----------------------------------------------------------------
	Errors_max_EE = log10(abs(Errors[discard:,0:(N)]).max()) 
	#Errors_max_EE = log10(max(max(abs(Errors(1+discard:end,1:N)))));    
	# Across N Euler equations
	
	Errors_max_MUC = log10(abs(Errors[discard:,N:(2*N)]).max()) 
	#Errors_max_MUC = log10(max(max(abs(Errors(1+discard:end,N+1:2*N)))));    
	# Across N optimality conditions w.r.t. consumption (conditions on   
	# marginal utility of consumption, MUC)
	
	Errors_max_MUL = log10(abs(Errors[discard:,2*N:(3*N)]).max()) 
	#Errors_max_MUL = log10(max(max(abs(Errors(1+discard:end,2*N+1:3*N)))));    
	# Across N optimality conditions w.r.t. labor (conditions on marginal 
	# utility of labor, MUL)
	
	Errors_max_RC = log10(abs(Errors[discard:,3*N]).max()) 
	#Errors_max_RC = log10(max(max(abs(Errors(1+discard:end,3*N+1)))));    
	# In the aggregate resource constraint 
	
	# 7. Time needed to run the test
	#-------------------------------
	time_test = clock()-tic;  

	return Errors_mean, Errors_max, time_test	
	