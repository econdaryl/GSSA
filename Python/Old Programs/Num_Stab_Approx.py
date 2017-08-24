# This version: July 14, 2011. First version: August 27, 2009.
# -------------------------------------------------------------------------
# Inputs:  "X" is a matrix of dependent variables in a regression, T-by-n,
#          where n corresponds to the total number of coefficients in the 
#          original regression (i.e. with unnormalized data);
#          "Y" is a matrix of independent variables, T-by-N; 
#          "RM" is the regression (approximation) method, RM=1,...,8:  
#          1=OLS,          2=LS-SVD,    3=LAD-PP,    4=LAD-DP, 
#          5=RLS-Tikhonov, 6=RLS-TSVD,  7=RLAD-PP,   8=RLAD-DP;
#          "penalty"  is a parameter determining the value of the regulari-
#          zation parameter for a regularization methods, RM=5,6,7,8;
#          "normalize"  is the option of normalizing the data, 
#          0=unnormalized data,  1=normalized data                
# 
# Outputs: "B" is a matrix of the regression coefficients 



from numpy import *
#from lpsolve55 import *
#from lp_maker import *
def Num_Stab_Approx(X,Y,RM,penalty,normalize):
#function B = Num_Stab_Approx(X,Y,RM,penalty,normalize)
	X = matrix(X)
	Y = matrix(Y)
# 1. Compute the dimensionality of the data
#------------------------------------------
	T = X.shape[0] #[T,n] = size(X);       % Compute the size of X; n corresponds to the total 
	n = X.shape[1]                       # number of regression coefficient to be computed
	N = Y.shape[1] #N = size(Y,2);         # Compute the number of columns in Y; N corresponds 
						# to the total number of regressions to be ran 

	# 2. Normalize the data
	#----------------------
	if (normalize == 1) or (RM>=5): #if (normalize == 1)||(RM>=5); 
						# If we use normalized data or consider a 
						# regularization method, ... 
		X1 = (X[:,1:n]-matrix(ones((T,1)))*mean(X[:,1:n],axis=0))/(matrix(ones((T,1)))*std(X[:,1:n],axis=0,ddof=1,dtype=float64)) #X1 = (X(:,2:n)-ones(T,1)*mean(X(:,2:n)))./(ones(T,1)*std(X(:,2:n)));
						# Center and scale X
		Y1 = (Y-matrix(ones((T,1)))*mean(Y,axis=0))/(matrix(ones((T,1)))*std(Y,axis=0,ddof=1,dtype=float64)) #Y1 = (Y-ones(T,1)*mean(Y))./(ones(T,1)*std(Y));  % Center and scale Y
		n1 = n-1 #n1 = n-1;          # Number of coefficients in a regression with 
						# normalized data is reduced by 1 (no intercept)

	else:                   # If we use unnormalized data, ...
		X1 = X           # Leave X without changes
		Y1 = Y            # Leave Y without changes
		n1 = n            # Leave n without changes 
	
	# 3. Regression methods
	#----------------------
	
	# 3.1 OLS
	#--------
	if RM == 1:               # If the regression method is OLS, ...
		#B = inv(X1'*X1)*X1'*Y1
		B = dot(dot(linalg.inv(dot(X1.T,X1)),X1.T),Y1)					# Compute B using the formula of the OLS
							# estimator; note that all the regressions,
							# j=1,...,N, are ran at once
		
	# 3.2 LS-SVD
	#-----------
	elif RM == 2:           # If the regression method is LS-SVD, ...
		U,S,V = linalg.svd(X1,0) #[U,S,V] = svd(X1,0);  # Compute an SVD of X1 using option "0", which is 
							# "economy size" SVD in MATLAB; matrices U, V and 
							# S are defined by X1=U*S*V', where U is T-by-n1, 
							# S is n1-by-n1, and V is n1-by-n1 
		U = matrix(U)
		V = matrix(V)
		
		S_inv = diag(1./S) #S_inv = diag(1./diag(S(1:n1,1:n1))); 
							# Compute an inverse of S
		B = dot(dot(dot(V.T,S_inv),U.T),Y1)   
		#B = V*S_inv*U'*Y1; % Compute B using the formula of the LS-SVD 
							# estimator (20) in JMM (2011)
							
	# For each regression j, RM 3, 4, 7, 8, we  solve a linear programming 
	# problem written in MATLAB:
	#                       min  f'*xlp 
	#                       s.t. Aeq*xlp=beq       (*)
	#                            Aineq*xlp<=bineq  (**) 
	#                            LB=xlp<=UB
	# where xlp is the vector of unknowns; Aeq is the matrix of coefficients 
	# in the equality restriction (*); Aineq is the matrix of coefficients in the
	# inequality restriction (**); beq and bineq are vectors of constants in the 
	# right sides of (*) and (**), respectively; LB means "lower bound", UB  
	# means "upper bound"
	
	# 3.3 LAD-PP
	#-----------
	#elif RM == 3:          # If the regression method is LAD-PP, ...
	#	# We solve the linear programming problem (27)-(29) in JMM (2011). 
	#	# xlp=[B(:,j); ups_plus; ups_minus] where B(:,j) is the vector of coeffi-
	#	# cients in the regression j, ups_plus and ups_minus are the deviations; 
	#	# f'=[0,...,0,1,...,1] where zeros correspond to the coefficients in the 
	#	# objective function on B and ones correspond to the coefficients on 
	#	# ups_plus and ups_minus
	#	LB = concatenate((zeros((n1,1))-100, zeros((2*T,1)))) #LB = [zeros(n1,1)-100; zeros(2*T,1)]; 
	#							# Lower bound on B is set to a small number, 
	#							# -100, and lower bounds on ups_plus and ups_minus 
	#							# are set to zero
	#	UB = concatenate((zeros((n1,1))+100, inf(2*T,1))) #ZZZ what is inf? #UB = [zeros(n1,1)+100; inf(2*T,1)]; 
	#							# Upper bound on B is set to a large number, 
	#							# 100, and upper bounds on ups_plus and ups_minus 
	#							# are set to infinity
	#	f = concatenate((zeros((n1,1)), ones((2*T,1)))) #f = [zeros(n1,1); ones(2*T,1)]; # (n1+2T)-by-1 
	#	Aeq =  concatenate((X1, eye(T,T), -eye(T,T)),1) #Aeq =  [X1 eye(T,T) -eye(T,T)]; # T-by-(n1+2T)
	#	
	#	B = zeros((X1.shape[1],N)) #B = zeros(size(X1,2),N); # Allocate memory for the matrix B, which will 
	#								# contain coefficients of all regressions; n1-by-N
	#	for j in range(1,N+1) #for j = 1:N              # For each regression j, ...
	#		beq = Y[:,j-1] #beq = Y1(:,j)       # T-by-1
	#		lp = lp_maker(f, [], [], [-1, -1, -1], l, u, None, 1, 0) # ZZZ figure out how to do this[xlp,fval,exitflag,output,lambda] = linprog(f,[],[],Aeq,beq,LB,UB,[]);
	#								# Find xlp 
	#		B[:,j-1] = xlp[0:n1,0]#B(:,j) = xlp(1:n1,1);# Store the regression coefficients for the 
	#								# regression j, xlp(1:n1,1), into the matrix B 
	#
	#
	# 3.4 LAD-DP
	#-----------
	#elif RM == 4:          # If the regression method is LAD-DP, ...
	#	# We solve the linear programming problem (30)-(32) in JMM (2011). 
	#	# xlp=[q] where q is a vector of unknowns in (30)-(32) of JMM (2011)
	#	LB = -ones((1,T)) #LB = -ones(1,T);      # Lower bound on q is -1
	#	UB = ones((1,T))  #UB = ones(1, T);      # Upper bound on q is 1
	#	Aeq = X1.T        #Aeq =  X1';           # n1-by-T    
	#	beq = zeros((n1,1)) #beq = zeros(n1,1);    # n1-by-1
	#	B = zeros((x1.shape[1],N)) #B = zeros(size(X1,2),N);
	#							# Allocate memory for the regression coefficients;
	#							# n1-by-N
	#	for j in range(1,N+1) #for j = 1:N           # For each regression j, ...
	#		f = -Y1[:,j-1] #f = -Y1(:,j);     # Our objective function is -Y1'(:,j)*q (minus sign 
	#							# appears because (30)-(32) in JMM (2011) is a 
	#							# minimization problem)
	#		xlp,fval,exitflag,output,lamb = linprog(f,[],[],Aeq,beq,LB,UB,[]) # ZZZ figure out how to do this [xlp,fval,exitflag,output,lambda] = linprog(f,[],[],Aeq,beq,LB,UB,[]);
	#							# Find the values of the Lagrange multipliers, 
	#							# lambda, on all the constraints
	#		B[:,j-1] = lamb.eqlin #ZZZ figure out how to do thisB(:,j) = lambda.eqlin;
	#							# Coefficients for the regression j, B(:,j),
	#							# are equal to the Lagrange multipliers on the  
	#							# equality constraint (*)
	#	
	#
	# 3.5 RLS-Tikhonov
	#-----------------   
	elif RM == 5: #elseif RM == 5;          # If the regression method is RLS-Tikhonov, ...  
		Btemp=linalg.inv(dot(X1.T,X1)+double(T)/double(n1)*eye(n1)*power(10.,penalty)) #B = inv(X1'*X1+T/n1*eye(n1)*10^penalty)*X1'*Y1;
		B = dot(dot(Btemp,X1.T),Y1)						# Use the formula (22) in JMM (2011) where the 
								# regularization parameter is T/n1*10^-penalty
		
	# 3.6 RLS-TSVD
	#-------------
	elif RM == 6: #elseif RM == 6           % If the regression method is RLS-TSVD, ...  
		[U,S,V]=linalg.svd(X1,0) #[U,S,V] = svd(X1,0);
							# Compute an SVD of X1 using option "0", 
							# which is "economy size" SVD in MATLAB; matrices 
							# U, V and S are defined by X1=U*S*V', where U is 
							# T-by-n1, S is n1-by-n1, and V is n1-by-n1  
		U = matrix(U)
		V = matrix(V)
		r = sum((max(S)/S)<=power(10.,penalty))	#r = sum((max(diag(S))./diag(S))<=10^penalty); 
							# Compute the number of singular values that are
							# <= than a threshold level equal to 10^penalty 
		Sr_inv = matrix(zeros((n1,n1))) 	#Sr_inv = zeros(n1); Sr_inv(1:r,1:r) = diag(1./diag(S(1:r,1:r)));
		Sr_inv[0:r,0:r] = matrix(diag(1/S[0:r]))					
								# Compute an inverse of the truncated matrix S
		B = dot(dot(dot(V.T,Sr_inv),U.T),Y1)	#B = V*Sr_inv*U'*Y1;
							# Compute the coefficients using the formula for  
							# the RLS-TSVD estimator (43) in JMM (2011)
	
	# 3.7 RLAD-PP
	#------------
	#elif RM == 7 #elseif RM == 7;          % If the regression method is RLAD-PP, ...  
	#	# We solve the linear programming problem (34)-(37) in JMM (2011). 
	#	# xlp=[phi_plus; phi_minus; ups_plus; ups_minus] where phi_plus, phi_minus, ups_plus, ups_minus are defined in 
	#	# (34)-(37) of JMM (2011) 
	#	LB = concatenate((zeros((2*n1,1)), zeros((2*T,1))),1) #LB = [zeros(2*n1,1); zeros(2*T,1)];
	#							# Lower bounds on phi_plus, phi_minus, ups_plus, ups_minus are 0
	#	#UB = [];              # No upper bounds at all
	#	f = concatenate((dot((power(10,penalty),ones((n1*2,1)),T/double(n1), ones((2*T,1)))))) #f = [10^penalty*ones(n1*2,1)*T/n1; ones(2*T,1)];
	#							# Regularization parameter is T/n1*10^-penalty; 
	#							# (2*n1+2T)-by-1
	#	Aeq =  concatenate((X1, -X1, eye(T,T), -eye(T,T)),1) #Aeq =  [X1 -X1 eye(T,T) -eye(T,T)];
	#							# T-by-(2*n1+2*T)
	#	B = zeros((X1.shape[1],N)) #B = zeros(size(X1,2),N);
	#							# Allocate memory for the regression coefficients;
	#							# n1-by-N
	#	for j in range(1,N+1)#for j = 1:N           % For each regression j, ...
	#		beq = Y[:,j-1] #beq =  Y1(:,j);   % T-by-1
	#		xlp,fval,exitflag,output,lamb = linprog(f,[],[],Aeq,beq,LB,UB,[]) #ZZZ figure out how to do this[xlp,fval,exitflag,output,lambda] = linprog(f,[],[],Aeq,beq,LB,UB,[]);
	#							# Find xlp in which xlp(1:n1,1) corresponds to phi_plus 
	#							# and xlp(n1+1:2*n1,1) corresponds to phi_minus
	#		B[:,j-1] = xlp[0:n1,0:1]-xlp[n1:(2*n1),0:2) #B(:,j) = xlp(1:n1,1)-xlp(n1+1:2*n1,1);
	#							# Coefficients for the regression j, B(:,j),
	#							# are given by the difference phi_plus - phi_minus;   
	#							# store these coefficients into the matrix B 
    #
	#
	# 3.8 RLAD-DP
	#------------
	#elif RM == 8:#elseif RM == 8;          % If the regression method is RLAD-DP, ...
	#	# We solve the linear programming problem (38)-(41) in JMM (2011). 
	#	# xlp=[q] where q is a vector of unknowns in (38)-(41) of JMM (2011) 
	#	LB = -ones((1,T))#LB = -ones(1,T);      % Lower bound on q is -1
	#	UB = ones((1,T)) #UB = ones(1,T);       % Upper bound on q is 1
	#	Aineq = concatenate((X1.T, -X1.T),1)  # Aineq = [X1'; -X1'];  % 2*n1-by-T
	#	bineq = dot((power(10,penalty),ones((n1*2,1)),T/double(n1)))  #bineq = 10^penalty*ones(n1*2,1)*T/n1;
	#							# *T/n1*10^-penalty is a regularization parameter;
	#							# 2*n1-by-1
	#	B = zeros((X1.shape[1],N))#B = zeros(size(X1,2),N);
	#							# Allocate memory for the regression coefficients;
	#							# n1-by-N
	#	for j in range(1,N+1): #for j = 1:N           # For each regression j, ...
	#		f = -Y1[:,j:(j+1)] #f = -Y1(:,j);     # Our objective function is -Y1'(:,j)*q (minus sign 
	#							# appears because (38)-(41) in JMM (2011) is a 
	#							# minimization problem)
	#		[xlp,fval,exitflag,output,lam] = linprog(f,Aineq,bineq,[],[],LB,UB,[]); #ZZZ lin program
	#							# Find the values of the Lagrange multipliers, 
	#							# lambda, on all the constraints; phi_plus and phi_minus 
	#							# (defined in (38)-(41) in JMM, 2011) are equal to  
	#							# the Lagrange multipliers lambda.ineqlin(1:n1) 
	#							# and lambda.ineqlin(n1+1:2*n1), respectively
	#		B[:,j:(j+1)] = lam.ineqlin(1:n1)-lamb.ineqlin(n1+1:2*n1) #B(:,j) = lambda.ineqlin(1:n1)-lambda.ineqlin(n1+1:2*n1);
	#							# Coefficients for the regression j, B(:,j),
	#							# are given by the difference phi_plus - phi_minus
	#	
	#
	#
	#
	# 10. Infer the regression coefficients in the original regression with
	# unnormalized data
	#----------------------------------------------------------------------
	if (normalize == 1)or(RM>=5): #if (normalize == 1)||(RM>=5);
							# If data were normalized, ... 
		a = matrix(1./std(X[:,1:n],axis=0,ddof=1,dtype=float64).T)
		b = matrix(std(Y,axis=0,ddof=1,dtype=float64))
		Btemp2 = multiply(dot(a,b),B) #B(2:n,:) = (1./std(X(:,2:n))')*std(Y).*B; 
							# Infer all the regression coefficients except 
							# of the intercept
		Btemp1 = mean(Y,axis=0)-dot(mean(X[:,1:n],axis=0),Btemp2) #B(1,:) = mean(Y)-mean(X(:,2:n))*B(2:n,:); 
							# Infer the intercept

		B = concatenate((Btemp1,Btemp2),0)
	return B
	