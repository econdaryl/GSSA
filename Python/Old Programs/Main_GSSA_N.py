
# ------------------------------------------------------------------------
# The software uses the following files: 
# ------------------------------------------------------------------------
# 1. "Main_GSSA_N.m"      computes GSSA solutions to the N-country model
# 2. "Accuracy_Test_N.m"  computes approximation errors in the optimality 
#                         conditions on a given set of points in the state 
#                         space, for the N-country model 
# 3. "Productivity.m"     generates random draws of the productivity shocks  
#                         and simulates the corresponding series of the  
#                         productivity levels
# 4. "Num_Stab_Approx.m"  implements the numerically stable LS and LAD 
#                         approximation methods
# 5. "Ord_Polynomial_N.m" constructs the sets of basis functions for ordinary 
#                         polynomials of the degrees from one to five, for
#                         the N-country model
# 6. "Monomials_1.m"      constructs integration nodes and weights for an N-
#                         dimensional monomial (non-product) integration rule 
#                         with 2N nodes 
# 7. "Monomials_2.m"      constructs integration nodes and weights for an N-
#                         dimensional monomial (non-product) integration rule 
#                         with 2N^2+1 nodes
# 8. "GH_Quadrature.m"    constructs integration nodes and weights for the 
#                         Gauss-Hermite rules with the number of nodes in 
#                         each dimension ranging from one to ten                     
# 9. "aT20200N10.mat"     contains the series of the productivity levels of 
#                         length 20,200 for 10 countries that are used for 
#                         computing solutions and for evaluating accuracy 


from numpy import *
from time import clock
from GSSA import *
from Num_Stab_Approx import *
from scipy import *

# 1. Choose the number of countries and simulation length
# -------------------------------------------------------
N=2 #N = 2;       # Choose the number of countries, 1<=N<=10 (note that the 
				 # code also works for the one-country case, N=1)
T = 150 #T = 2000;    # Choose the simulation length for the solution procedure,
				 # T<=10,000   
				 
# To solve models with N>10 or T>10,000, one needs to simulate new series
# of the productivity levels by enabling the code in paragraph 6.  

# 2. Model's parameters
# ---------------------
gam     = float(1);      # Utility-function parameter
alpha   = float(0.36);   # Capital share in output
beta    = float(0.99);   # Discount factor
delta   = float(0.025);  # Depreciation rate 
rho     = float(0.95);   # Persistence of the log of the productivity level
sigma   = float(0.01);   # Standard deviation of shocks to the log of the 
				  # productivity level
vcv = (sigma**2)*matrix((eye(N)+ones((N,N),dtype=float))) #vcv = sigma^2*(eye(N)+ones(N,N)); 

				  # Variance-covariance matrix of the countries' productivity 
				  # shocks in which diagonal terms are equal to 2*sigma^2   
				  # and in which off-diagonal terms are equal to sigma^2; 
				  # this vcv follows from the assumption that a country's 
				  # shock has both common-for-all-countries and country-
				  # specific components; N-by-N

							  
# 3. The normalizing constant, A, and welfare weight, tau
# -------------------------------------------------------
A       = (1-beta+beta*delta)/alpha/beta #A       = (1-beta+beta*delta)/alpha/beta;  # The normalizing constant in output  
tau = 1 #tau     = 1;                               # The welfare weight of country j 

# The above normalization ensures that steady state of capital of all 
# countries is equal to one 

# 4. Initial condition
# --------------------
k = matrix(ones((T+1,N),dtype = float)) #k(1,1:N) = 1;  # Initial condition for capital (is equal to steady state)
a = matrix(ones((1,N),dtype = float)) #a(1,1:N) = 1;  # Initial condition for the productivity level (is equal to 
x = matrix(zeros((T,2*N+1),dtype = float))               # steady state)

# 5. Construct the productivity levels, a 
# ---------------------------------------
a20200 = Productivity(T,N,a[0,0:N],sigma,rho);
							   # Generate a random draw of the productivity 
							   # shocks and simulate the corresponding  
							   # series of the productivity levels of length   
							   # T periods for N countries 
# save aT20200N10 a20200;       # Save the series of the productivity levels  
							   # into a file "aT20200N10.mat" 
#load aT20200N10;               # Load the previously saved series of the 
							   # productivity levels of length 20,200 for 
							   # 10 countries (the first 10,000 observations
							   # are used for finding a solution, and the 
							   # remaining 10,200 observations are used for
							   # evaluating accuracy)
f = open('a20200.txt','r')
a20200 = matrix(f.read()).reshape(20200,10)
a = matrix(a20200[0:T,0:N])  #a = a20200(1:T,1:N);           # Restrict the series of the productivity 
							   # levels for the solution procedure to the 
							   # given T<=10,000 and N<=10


# _________________________________________________________________________
#                               
# Compute a first-degree polynomial solution using the one-node Monte Carlo  
# integration method (this solution will be used as an initial guess for
# the other cases) 
# _________________________________________________________________________
#
tic = clock() #tic;           # Start counting time needed to compute the solution
							
# 6. The GSSA parameters  
# ---------------------
kdamp = float(0.1) #kdamp     = 0.1;     # Damping parameter for (fixed-point) iteration on 
					 # the coefficients of the capital policy functions
dif_1d = float(1e+10) #dif_1d = 1e+10;      # Set the initial difference between the series from
					 # two iterations in the convergence criterion (condition
					 # (10) in JMM, 2011) to a very large number
# To achieve convergence under N>10, one may need to modify the values of 
# the damping parameter kdamp or refine the initial guess 

# 7. Initialize the first-degree capital policy functions of N countries 
#-----------------------------------------------------------------------  
temp1 = matrix(zeros((1,N),dtype=float),copy=False)  
temp2 = matrix(diag(ones((N),dtype=float)*0.9),copy=False)   
temp3 = matrix(diag(ones((N),dtype=float)*0.1),copy=False)       
bk_1d  = bmat('temp1;temp2;temp3')

# bk_1d  = [zeros(1,N); diag(ones(1,N)*0.9);diag(ones(1,N)*0.1)]; 
# Matrix of polynomial coefficients of size (1+2N)-by-N: for each country  
# (each column), 1+2N rows correspond to a constant, N coefficients on the
# capital stocks, k(t,1),...,k(t,N), and N coefficients on the productivity 
# levels, a(t,1),...,a(t,N)

# As an initial guess, assume that a country's j capital depends only on 
# its own capital and productivity level as k(t+1,j)=0.9*k(t,j)+0.1*a(t,j); 
# (note that in the steady state, we have k(t+1,j)=0.9*k(t,j)+0.1*a(t,j)=1)

# Note that diag(ones(1,N)*q) delivers an N-by-N matrix with  diagonal 
# entries equal to q. 
#
# 8. Initialize the capital series
# --------------------------------
k_old = matrix(ones((T+1,N),dtype = float),copy=False)  #k_old = ones(T+1,N);  # Initialize the series of next-period capital of N
					  # countries; these series are used to check the
					  # convergence on the subsequent iteration (initially, 
					  # capital can take any value); (T+1)-by-N
					   
# 9. The main iterative cycle of GSSA
# -----------------------------------              
while dif_1d > 1e-4*kdamp: #while dif_1d > 1e-4*kdamp;        # 10^4*kdamp is a convergence parameter,
								# adjusted to the damping parameter; see 
								# JMM (2011) for a discussion
	
	# 9.1 Generate time series of capital
	# -----------------------------------
	

	for t in range(1,T+1): #for t = 1:T 
		temp1 = matrix(1)
		temp2 = k[t-1,:]
		temp3 = a[t-1,:]
		x[t-1,:] = bmat('temp1 temp2 temp3')
		#x(t,:) = [1 k(t,:) a(t,:)];   # The basis functions of the first-degree 
									# polynomial of at time t
		k[t,:] = x[t-1,:]*bk_1d;
		#k(t+1,:) = x(t,:)*bk_1d;      # Compute next-period capital using bk_1d
	#end
	
	#for t in range(1,T+1): #for t = 1:T 
	#	temp1 = matrix(1)
	#	temp2 = k[t-1,:]
	#	temp3 = a[t-1,:]
	#	x[t-1,:] = bmat('temp1 temp2 temp3')
	#	#x(t,:) = [1 k(t,:) a(t,:)];   # The basis functions of the first-degree 
	#								# polynomial of at time t
	#	expr = "k[t,:] = x[t-1,:]*bk_1d"
	#	blitz(expr,check_size=0)
	#	#k(t+1,:) = x(t,:)*bk_1d;      # Compute next-period capital using bk_1d
	##end

	#code = """
	#		for(int i=0; i<(T),i++) {
	#			k(i+1,0) = 1*bk_1d(0,0);			
	#			for(int j =0; j<N,j++) {
	#				k(i+1,j+1) = k(i,j) *bk_1d(j+1,0);
	#				k(i+1,j+N+1) = a(i,j) *bk_1d(j+N+1,0);
	#			}
	#		}
	#		return k
	#		"""
	#k = inline(code,['k','a','bk_1d','T','N'], type_converters=converters.blitz, compiler='gcc')
	# 9.2 Compute consumption series 
	#-------------------------------
	C = (A*multiply(power(k[0:T,:],alpha),a[0:T,:]) - k[1:T+1,:]+k[0:T,:]*(1-delta))*ones((N,1),dtype=float)
	#C = (A*k(1:T,:).^alpha.*a(1:T,:) - k(2:T+1,:)+k(1:T,:)*(1-delta))*ones(N,1);
	# Aggregate consumption is computed by summing up individual consumption, 
	# which in turn, is found from the individual budget constraints; T-by-1
	
	c = C*ones((1,N),dtype=float)/float(N) #c = C*ones(1,N)/N;                # Individual consumption is the same for
									# all countries; T-by-N 
	#problem here ZZZ
	# 9.3 Evaluate the percentage (unit-free) difference between the series  
	# from the previous and current iterations
	# ---------------------------------------------------------------------
	
	
	dif_1d = mean(absolute(1-k/k_old),dtype=float)
	#dif_1d = mean(mean(abs(1-k./k_old)))
					# Compute a unit-free difference between the capital series 
					# from two iterations; see condition (10) in JMM (2011)
				
	# 9.4 Monte Carlo realizations of the right side of the Euler equation, Y, 
	# in condition (C4) in the online Appendix C
	#-------------------------------------------------------------------------
	Y = matrix(zeros((T,N),dtype = float),copy=False)
	for j in range(1,N+1): #for j = 1:N
		temp1 = beta*power(c[1:T,j-1],-gam)/power(c[0:T-1,j-1],-gam)
		temp2 = 1-delta+alpha*A*multiply(power(k[1:T,j-1],(alpha-1)),a[1:T,j-1])
		Y[0:T-1,j-1] =multiply(multiply(temp1,temp2),k[1:T,j-1]);
		#Y(1:T-1,j) = beta*c(2:T,j).^(-gam)./c(1:T-1,j).^(-gam).*(1-delta+alpha*A*k(2:T,j).^(alpha-1).*a(2:T,j)).*k(2:T,j);
	#end  % (T-1)-by-N
	
	# 9.5 Compute and update the coefficients of the capital policy functions 
	# -----------------------------------------------------------------------
	bk_hat_1d  = (x[0:T-1,:].T*x[0:T-1,:]).I*x[0:T-1,:].T*Y[0:T-1,:];    
	#bk_hat_1d  = inv(x(1:T-1,:)'*x(1:T-1,:))*x(1:T-1,:)'*Y(1:T-1,:);    
								# Compute new coefficients of the capital 
								# policy functions using the OLS
	bk_1d = kdamp*bk_hat_1d + (1-kdamp)*bk_1d; 
	#bk_1d = kdamp*bk_hat_1d + (1-kdamp)*bk_1d; 
								# Update the coefficients of the capital  
								# policy functions using damping 
										
	# 9.6 Store the capital series 
	#-----------------------------
	k_old = k.copy() #k_old = k;         # The stored capital series will be used for checking 
					#the convergence on the subsequent iteration
#end;

# 10. Time needed to compute the initial guess 
# --------------------------------------------
time_GSSA_1d = clock()-tic
#time_GSSA_1d     = toc; 
# _________________________________________________________________________                              
#
# Compute polynomial solutions of the degrees from one to D_max using one 
# of the following integration methods: Monte Carlo, Gauss-Hermite product  
# and monomial non-product methods
# _________________________________________________________________________

tic = clock() #tic;                  # Start counting time needed to compute the solution
							
# 11. The GSSA parameters  
# -----------------------
kdamp = 0.1 #kdamp     = 0.1;      # Damping parameter for (fixed-point) iteration on 
					  # the coefficients of the capital policy functions
dif_GSSA_D  = 1e+10 #dif_GSSA_D  = 1e+10;  # Set the initial difference between the series from
					  # two iterations in the convergence criterion (condition
					  # (10) in JMM, 2011) to a very large number

# To achieve convergence under N>10, one may need to modify the values of 
# the damping parameter kdamp or refine the initial guess 

# 12. The matrix of the polynomial coefficients
# ---------------------------------------------                             
D_max = 5 #D_max  = 5;           # Maximum degree of a polynomial: the program computes
					  # polynomial solutions of the degrees from one to D_max;
					  # (D_max can be from 1 to 5) 
npol = zeros((1,D_max))
for D in range(1,D_max+1): #for D = 1:D_max         # For the polynomial degrees from one to D_max
	npol[0,D-1] = Ord_Polynomial_N(concatenate((k[0,:], a[0,:]),1),D).shape[1]      
	#npol(D) = size(Ord_Polynomial_N([k(1,:) a(1,:)],D),2);      
#end
					  # Construct polynomial bases and compute their number
					  # (this is needed for finding the number of the polynomial
					  # coefficients in the policy functions)
					  
BK = zeros((npol[0,D_max-1],N,D_max)) #BK = zeros(npol(D_max),N,D_max); # Matrix of polynomial coefficients of the 
								 # capital policy functions for the polynomial
								 # solutions of the degrees from one to D_max;
								 # npol(D_max)-by-N-by-D_max
								 
# 13. Choose an integration method 
# --------------------------------                             
IM    = 11;      # 0=a one-node Monte Carlo method(default);
				 # 1,2,..,10=Gauss-Hermite quadrature rules with 1,2,...,10 
				 # nodes in each dimension, respectively;
				 # 11=Monomial rule with 2N nodes;
				 # 12=Monomial rule with 2N^2+1 nodes
if (IM>=1) and (IM<=10): #if (IM>=1)&&(IM<=10)
	n_nodes,epsi_nodes,weight_nodes = GH_Quadrature(IM,N,vcv);
							# Compute the number of integration nodes, 
							# n_nodes, integration nodes, epsi_nodes, and 
							# integration weights, weight_nodes, for Gauss-
							# Hermite quadrature integration rule with IM 
							# nodes in each dimension
elif IM == 11:
	[n_nodes,epsi_nodes,weight_nodes] = Monomials_1(N,vcv);
							# Monomial integration rule with 2N nodes
elif IM == 12:
	[n_nodes,epsi_nodes,weight_nodes] = Monomials_2(N,vcv);
							# Monomial integration rule with 2N^2+1 nodes
#end    

# Under the one-node Gauss-Hermite quadrature rule, the conditional 
# expectation (integral) is approximated by the value of the integrand 
# evaluated in one integration node in which the next-period productivity 
# shock is zero, i.e., the next-period productivity level is 
# a(t+1,:)=a(t,:).^rho*exp(0)=a(t,:).^rho

# 14. Choose a regression method 
# ------------------------------                             
RM    = 1;       # Choose a regression method: 
				 # 1=OLS,          2=LS-SVD,   3=LAD-PP,  4=LAD-DP, 
				 # 5=RLS-Tikhonov, 6=RLS-TSVD, 7=RLAD-PP, 8=RLAD-DP
normalize = 0;   # Option of normalizing the data; 0=unnormalized data; 
				 # 1=normalized data                    
penalty = 7;     # Degree of regularization for a regularization methods, 
				 # RM=5,6,7,8 (must be negative, e.g., -7 for RM=5,7,8 
				 # and must be positive, e.g., 7, for RM=6)

# 15. Compute the polynomial solutions of the degrees from one to D_max
# ---------------------------------------------------------------------
time_GSSA = zeros((1,D_max),dtype=float)
for D in range(1,D_max): #for D=1:D_max
	
	# 15.1 Using the previously computed capital series, compute the initial 
	# guess on the coefficients under the  selected approximation method
	# ----------------------------------------------------------------------
	X = Ord_Polynomial_N(concatenate((k[0:T,:],a[0:T,:]),1),D) 
	#X = Ord_Polynomial_N([k(1:T,:) a(1:T,:)],D);   
						# Construct the polynomial bases on the series 
						# of state variables from the previously computed 
						# time-series solution
	bk_D = Num_Stab_Approx(X[0:T-1,:],Y[0:T-1,:],RM,penalty,normalize);
	#bk_D = Num_Stab_Approx(X(1:T-1,:),Y(1:T-1,:),RM,penalty,normalize);
						# Compute the initial guess on the coefficients
						# using the chosen regression method
	k_old = matrix(ones((T+1,N),dtype=float),copy=False)    # Initialize the series of next-period capital of N
	#k_old = ones(T+1,N);  # countries; these series are used to check the
						# convergence on the subsequent iteration (initially, 
						# capital can take any value); (T+1)-by-N
	dif_GSSA_D  = 1e+10;   # Convergence criterion (initially is not satisfied)
	

	# 15.2 The main iterative cycle of GSSA
	# -------------------------------------              
	while dif_GSSA_D > 1e-4/(10.**D)*kdamp: #while dif_GSSA_D > 1e-4/10^D*kdamp;   
								# 10^(-4-D)*kdamp is a convergence parameter, 
								# adjusted to the polynomial degree D and the 
								# damping parameter kdamp; see the discussion in 
								# JMM (2011)
		
		# 15.2.1 Generate time series of capital
		#	--------------------------------------
		for t in range(1,T+1): #for t = 1:T     
			X[t-1,:] = Ord_Polynomial_N(concatenate((k[t-1,:],a[t-1,:]),1),D)
			#X(t,:) = Ord_Polynomial_N([k(t,:) a(t,:)],D);  
			# The basis functions of a polynomial of degree D at time t
			k[t,:] = X[t-1,:]*bk_D #k(t+1,:) = X(t,:)*bk_D;                      
			# Compute next-period capital using bk_D
		#end
		# 15.2.2 Compute consumption series of all countries 
		#---------------------------------------------------
		k0  =  k[0:T,:];   #k0  =  k(1:T,:);     # N current capital stocks  
		a0  =  a[0:T,:];   #a0  =  a(1:T,:);     # N current productivity levels  
		k1  =  k[1:T+1,:]; #k1  =  k(2:T+1,:);       # N next-period capital stocks 
		C = (A*multiply(power(k0,alpha),a0) - k1+k0*(1-delta))*ones((N,1),dtype=float) 
		#C = (A*k0.^alpha.*a0 - k1+k0*(1-delta))*ones(N,1);
		# Aggregate consumption is computed by summing up individual consumption, 
		# which in turn, is found from the individual budget constraints; T-by-1

		
		c = C*ones((1,N),dtype=float)/float(N) #c = C*ones(1,N)/N;           # Individual consumption is the same for
									# all countries; T-by-N 
		# 15.2.3 Approximate the conditional expectations for t=1,...T-1 using 
		# the integration method chosen
		#----------------------------------------------------------------------
		# 
		# 15.2.3.1 The one-node Monte Carlo integration method approximates the 
		# values of the conditional expectations, Y, in the Euler equation with
		# the realization of the integrand in the next period
		# ---------------------------------------------------------------------
		if IM == 0:#if IM == 0;
			for j in range(1,N+1): #for j = 1:N
				temp1 = beta*power(c[1:T,j-1],-gam)/power(c[0:T-1,j-1],-gam)
				temp2 = 1-delta+alpha*A*multiply(power(k[1:T,j-1],(alpha-1)),a[1:T,j-1])
				Y[0:T-1,j-1] = multiply(multiply(temp1,temp2),k[1:T,j-1]);
				#Y(1:T-1,j) = beta*c(2:T,j).^(-gam)./c(1:T-1,j).^(-gam).*(1-delta+alpha*A*k(2:T,j).^(alpha-1).*a(2:T,j)).*k(2:T,j);
			#end
		# 15.2.3.2 Deterministic integration methods approximate the values of 
		# conditional expectations, Y, in the Euler equation as a weighted average 
		# of the values of the integrand in the given nodes with the given weights 
		# ------------------------------------------------------------------------
		else: #else
			Y = matrix(zeros((T,N),dtype=float),copy=False) #Y = zeros(T,N); #Allocate memory for the variable Y
			for i in range(1,n_nodes+1): #for i = 1:n_nodes         
				a1  =  multiply(power(a[0:T,:],rho),exp(matrix(ones((T,1)),copy=False)*epsi_nodes[i-1,:])) 
				#a1  =  a(1:T,:).^rho.*exp(ones(T,1)*epsi_nodes(i,:));   
				# Compute the next-period productivity levels for each integration       
				# node using condition (C3) in the online Appendix C; n_nodes-by-N
					
				k2  =  Ord_Polynomial_N(concatenate((k1, a1),1),D)*bk_D #k2  =  Ord_Polynomial_N([k1 a1],D)*bk_D; 
				# Compute capital of period t+2 (chosen at t+1) using the
				# capital policy functions; n_nodes-by-N 
				
				C1 = (A*multiply(power(k1,alpha),a1) - k2+k1*(1-delta))*ones((N,1)) #C1 = (A*k1.^alpha.*a1 - k2+k1*(1-delta))*ones(N,1);
				# C is computed by summing up individual consumption, which in
				# turn, is found from the individual budget constraints; T-by-1
				
				c1 = C1*ones((1,N),dtype=float)/float(N) #c1 = C1*ones(1,N)/N;                 
				# Compute next-period consumption for N countries; n_nodes-by-N
				for j in range(1,N+1): #for j = 1:N
					#temp1 = weight_nodes[i-1,0]*beta*power(c1[0:T,j-1],-gam)/power(c[0:T,j-1],-gam)
					#temp2 = 1-delta+alpha*A*multiply(power(k1[0:T,j-1],alpha-1),a1[0:T,j-1])
					#Y[0:T,j-1] = Y[0:T,j-1]+ multiply(multiply(temp1,temp2),k1[0:T,j-1]);
					Y[0:T,j-1] = Y[0:T,j-1]+ multiply(multiply(weight_nodes[i-1,0]*beta*power(c1[0:T,j-1],-gam)/power(c[0:T,j-1],-gam),1-delta+alpha*A*multiply(power(k1[0:T,j-1],alpha-1),a1[0:T,j-1])),k1[0:T,j-1])
					#Y(1:T,j) = Y(1:T,j)+weight_nodes(i,1)*beta*c1(1:T,j).^(-gam)./c(1:T,j).^(-gam).*(1-delta+alpha*A*k1(1:T,j).^(alpha-1).*a1(1:T,j)).*k1(1:T,j);
				#end % T-by-N
			#end
		#end

		# 15.2.4 Evaluate the percentage (unit-free) difference between the 
		# capital series from the previous and current iterations
		# -----------------------------------------------------------------
		# 
		dif_GSSA_D = mean(abs(1-k/k_old)) #dif_GSSA_D = mean(mean(abs(1-k./k_old)))
					# Compute a unit-free difference between the capital series 
					# from two iterations; see condition (10) in JMM (2011)
		
		# 15.2.5 Compute and update the coefficients of the capital policy 
		# functions 
		# ----------------------------------------------------------------
		bk_hat_D = Num_Stab_Approx(X[0:T-1,:],Y[0:T-1,:],RM,penalty,normalize) 
		#bk_hat_D = Num_Stab_Approx(X(1:T-1,:),Y(1:T-1,:),RM,penalty,normalize); 
								# Compute new coefficients of the capital 
								# policy functions using the chosen 
								# approximation method
		bk_D = kdamp*bk_hat_D + (1-kdamp)*bk_D #bk_D = kdamp*bk_hat_D + (1-kdamp)*bk_D; 
								# Update the coefficients of the capital  
								# policy functions using damping 								
		# 15.2.6 Store the capital series 
		#--------------------------------
		k_old = k.copy() #k_old = k;       # The stored capital series will be used for checking 
						# the convergence on the subsequent iteration
	#end;
	
	# 15.2.7 The GSSA output for the polynomial solution of degree D
	# --------------------------------------------------------------
	
	BK[0:npol[0,D-1],0:N,D-1] = bk_D #BK(1:npol(D),1:N,D) = bk_D; #Store the coefficients of the polynomial  
								#of degree D that approximates capital 
								#policy functions of N countries 
	time_GSSA[0,D-1] = clock()-tic #time_GSSA(D) = toc;         #Time needed to compute the polynomial  
								#solution of degree D 
#end     
#f = open('bk.txt','w')
#f.write(str(BK))
#f.close()                         
# 16. Accuracy test of the GSSA solutions: errors on a stochastic simulation 
# --------------------------------------------------------------------------
#
# 16.1 Specify a set of points on which the accuracy is evaluated
#----------------------------------------------------------------
T_test = 225 #T_test = 10200;                  % Choose the simulation length for the test 
								 # on a stochastic simulation, T_test<=10,200 

a_test = a20200[T:T+T_test,0:N]  #a_test = a20200(T+1:T+T_test,1:N); # Restrict the series of the productivity 
								   # levels for the test on a stochastic 
								   # simulation to the given T_test<=10,200  
								   # and N<=10                             
		  
k_test = matrix(ones((T_test,2),dtype=float),copy=False)#k_test(1,1:N) = 1;  % Initial condition for capital (equal to steady state)

# 16.2 Choose an integration method for evaluating accuracy of solutions
#-----------------------------------------------------------------------
IM_test = 11 #IM_test = 11;      # See paragraph 13 for the integration 
								 # options

# To implement the test on a stochastic simulation with T_test>10,200, one
# needs to simulate new series of the productivity levels with larger T_test 
# by enabling the code in paragraph 6.

# 16.3 Compute errors on a stochastic simulation for the GSSA polynomial 
# solution of degrees D=1,...,D_max
# ----------------------------------------------------------------------
Errors_mean = matrix(zeros((1,D_max),dtype=float),copy=False)
Errors_max = matrix(zeros((1,D_max),dtype=float),copy=False)
time_test = matrix(zeros((1,D_max),dtype=float),copy=False)
for D in range(1,D_max+1): #for D = 1:D_max 
	
	# 16.3.1 Simulate the time series solution under the given capital-
	# policy-function coefficients, BK(:,:,D) with D=1,...,D_max 
	#------------------------------------------------------------------
	bk = BK[0:npol[0,D-1],:,D-1] #bk = BK(1:npol(D),:,D);     # The vector of coefficients of the 
								# polynomial of degree D 
	for t in range(1,T_test): #for t = 1:T_test-1
		X_test = Ord_Polynomial_N(concatenate((k_test[t-1,:], a_test[t-1,:]),1),D)
		#X_test = Ord_Polynomial_N([k_test(t,:) a_test(t,:)],D);
		# The basis functions of a polynomial of degree D at time t
		k_test[t,:] = X_test*bk #k_test(t+1,:) = X_test*bk;
		# Compute next-period capital using bk
	#end    
	
	# 16.3.2 Errors across 10,000 points on a stochastic simulation
	# -------------------------------------------------------------
	discard = 200 #discard = 200; # Discard the first 200 observations to remove the effect
				# of the initial conditions 
	Errors_mean[0,D-1],Errors_max[0,D-1], time_test[0,D-1] = Accuracy_Test_N(k_test,a_test,bk,D,IM_test,alpha,gam,delta,beta,A,tau,rho,vcv,discard);
	#[Errors_mean(1,D),Errors_max(1,D), time_test(1,D)] = Accuracy_Test_N(k_test,a_test,bk,D,IM_test,alpha,gam,delta,beta,A,tau,rho,vcv,discard);
	
	# Errors_mean    is the unit-free average absolute approximation error  
	#                across 4N+1 optimality conditions (in log10) 
	# Errors_max     is the unit-free maximum absolute approximation error   
	#                across 4N+1 optimality conditions (in log10) 
#end

# 17. Display the results for the polynomial solutions of the degrees from 
# one to D_max  
# ------------------------------------------------------------------------
print('RUNNING TIME (in seconds):')
print('a) for computing the solution') 
print repr(time_GSSA) +'\n'
print('b) for implementing the accuracy test') 
print repr(time_test) +'\n'
print('APPROXIMATION ERRORS (log10):')
print('a) mean error across 4N+1 optimality conditions'); 
print repr(Errors_mean)+'\n'
print('b) max error across 4N+1 optimality conditions'); 
print repr(Errors_max)+'\n'


#save Results_N time_GSSA time_test Errors_mean Errors_max kdamp RM IM N T BK k_test a_test IM_test alpha gam delta beta A tau rho vcv discard npol D_max T_test ;
