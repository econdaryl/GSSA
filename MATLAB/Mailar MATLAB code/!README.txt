 



The folder “GSSA_Two_Models” contains MATLAB software for generalized 
stochastic-simulation algorithm (GSSA) accompanying the article "Numerically 
Stable and Accurate Stochastic Simulation Approaches for Solving Dynamic 
Economic Models" by Kenneth L. Judd, Lilia Maliar and Serguei Maliar, 
published in Quantitative Economics (2011), 2/2, 173–210.  
 
This version: July 14, 2011. First version: August 27, 2009.

The following items are provided: 

I. LICENSE AGREEMENT.

II. Folder "GSSA_1_agent_Capital" contains MATLAB files that solve the one-agent model
    by parameterizing the capital policy function 

    1. "Main_GSSA_1.m"      computes GSSA solutions (in the form of capital 
                            policy function) to the standard one-agent model
    2. "Accuracy_Test_1.m"  computes Euler equation approximation errors on 
                            a given set of points in the state space, for the
                            standard one-agent model 
    3. "Num_Stab_Approx.m"  implements the numerically stable LS and LAD 
                            approximation methods 
    4. "Ord_Herm_Pol_1.m"   constructs the sets of basis functions for ordinary
                            and Hermite polynomials of the degrees from one 
                            to five, for the one-agent model  
    5. "GH_Quadrature.m"    constructs integration nodes and weights for 
                            Gauss-Hermite quadrature integration 
    6. "epsi10000.mat"      contains the series of the productivity shocks of 
                            length 10,000 that are used for computing solutions
    7. "epsi_test.mat"      contains the series of the productivity shocks of 
                            length 10,200 that are used for evaluating
                            accuracy. 

    To solve the model, execute "Main_GSSA_1.m".

III. Folder "GSSA_N_countries" contains MATLAB files that solve the N-country model

    1. "Main_GSSA_N.m"      computes GSSA solutions to the N-country model
    2. "Accuracy_Test_N.m"  computes approximation errors in the optimality 
                            conditions on a given set of points in the state 
                            space, for the N-country model 
    3. "Productivity.m"     generates random draws of the productivity shocks  
                            and simulates the corresponding series of the  
                            productivity levels
    4. "Num_Stab_Approx.m"  implements the numerically stable LS and LAD 
                            approximation methods
    5. "Ord_Polynomial_N.m" constructs the sets of basis functions for ordinary 
                            polynomials of the degrees from one to five, for
                            the N-country model
    6. "Monomials_1.m"      constructs integration nodes and weights for an N-
                            dimensional monomial (non-product) integration rule 
                            with 2N nodes 
    7. "Monomials_2.m"      constructs integration nodes and weights for an N-
                            dimensional monomial (non-product) integration rule 
                            with 2N^2+1 nodes
    8. "GH_Quadrature.m"    constructs integration nodes and weights for the 
                            Gauss-Hermite rules with the number of nodes in 
                            each dimension ranging from one to ten                     
    9. "aT20200N10.mat"     contains the series of the productivity levels of 
                            length 20,200 for 10 countries that are used for 
                            computing solutions and for evaluating accuracy 

    To solve the model, execute "Main_GSSA_N.m".

For updates and other related software, please, check the authors' web 
pages. For additional information, please, contact the corresponding 
author: Serguei Maliar, T24, Hoover Institution, 434 Galvez Mall, 
Stanford University, Stanford, CA 94305-6010, USA, maliars@stanford.edu.

-------------------------------------------------------------------------
Copyright © 2011 by Lilia Maliar and Serguei Maliar. All rights reserved. 
The code may be used, modified and redistributed under the terms provided 
in the file "License_Agreement.txt".
-------------------------------------------------------------------------