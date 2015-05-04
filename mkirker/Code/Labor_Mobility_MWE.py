# A MODEL OF PRODUCTIVITY DIFFUSION THROUGH LABOR MOBILITY
# ======================================================================================================================
# Model by Michael Kirker
#
# This code estimates the Balanced Growth Path (BGP) for my model of productivity diffusion. Documentation of the
# model's structure can be found at: [INSERT LINK HERE].
#
# ======================================================================================================================





# ======================================================================================================================
# LOAD RESOURCES AND PACKAGES
# ======================================================================================================================
# Import Python packages that will get used in the estimation of the model

import numpy as np
import matplotlib.pyplot as plt



# ======================================================================================================================
# PARAMETERIZATION OF THE MODEL
# ======================================================================================================================
# Here the user can input the parameterization of the model. Aside from the settings in this cell, no other
# user input is required in the rest of the code.



# Define the state-space grid of the model
#----------------------------------------------------------
# The continuous state space of productivity is approximated by a discrete grid

grid_size = 25        # Number of grid points
grid_max  = 5.0       # Maximium value in the grid
grid_min  = 0.0001    # Minimum value in the grid (a little bit above zero)


z_grid  = np.linspace(grid_min, grid_max, grid_size)    # Create grid
dz      = z_grid[1]-z_grid[0]                           # Record the grid step size



# Model Parameters
#----------------------------------------------------------
# Define the model's parameter values and store these in a dictionary ("para")

para = dict()   # Initialize container


para['alpha'] = 1.0/3.0   # Labor elasticity in production function
para['sigma'] = 5.0       # Elasticity of sub. between goods

para['N'] = 14.0          # Relative size of the labor force to the number of firms (a.k.a. avg. num of workers per firm)

para['r']     = 0.06      # Discount rate
para['gamma'] = 0.02      # Initial guess for growth rate



# Initial distribution of firm productivity:
para['delta_F'] = 0.2       # Tail parameter of firm productivity dist
para['lambda_F'] = 5.7e-03  # Scale parameter of firm productivity dist


# Manager negotiation parameters
para['beta'] = 0.05 # Worker's relative bargaining strength
para['tau']  = 0.1  # Fraction of time delay in finding internal candidate


# Search costs
para['psi']   = 35.0 # Cost coefficient in vacancy posting rate
para['psi_s'] = 35.0 # Cost coefficient in worker searching
# Works for 400, 300, 150 :: 100, 75, 50, 35 (after lowering q parameter)      25 DOESNT WORK


# Matching function parameters
para['q_norm'] = 0.25    # normalizing coefficient inside q(theta)
para['q_cd']   = 0.25  # Cobb-douglas parameter on v inside q(theta)




# Convergence criteria parameters
#----------------------------------------------------------

# Fudge factor (ff) = how much weight to give to the new draw (and 1-ff to the old draw) when updating the value of variables each iteration.
# Note, a lower "ff" value means the code takes longer to run, but is generally more stable in converging

ff_pol      = 0.1   # Fudge factor when updating policy rules
ff_vf       = 0.1   # Fudge factor when updating value functions
ff_phi      = 0.1   # Fudge factor when updating the productivity distribution and growth rate

# Covergence tolerance criteria = determines when to stop iterating because convergence has been reached.
tol_pol     = 0.0001    # How accurate convergence in policy rules must be
tol_vf      = 0.0001    # How accurate convergence in value functions must be
tol_phi     = 0.0001    # How accurate convergence in distribution must be


# Maximum number of iterations to try
max_pol_iter    = 300   # Maximum number of iterations to try before we declare policy code does not converge
max_vf_iter     = 300   # Maximum number of iterations to try before we declare value function code does not converge
max_phi_iter    = 300   # Maximum number of iterations to try before we declare productivity dist code does not converge

max_main_iter   = 300

print_skip = 10 # Print the convergence progress for every `print_skip'-th iteration






# ======================================================================================================================
#   OTHER PARAMETERIZATIONS ETC
# ======================================================================================================================
# Based on the user input above, define other model parameters and distributions that are a function of the parameters
# already defined.


# Construct the initial guess for the firm productivity distribution based on the Frechet distribution
phi_F =  para['lambda_F'] / para['delta_F'] * z_grid ** ( -1.0/para['delta_F'] - 1.0 )          \
         * np.exp( -para['lambda_F'] * z_grid ** ( - 1/para['delta_F'] ) )


# Renormalize scale parameter
para['lambda_F'] = para['lambda_F'] / sum(phi_F)

phi_F = phi_F / sum(phi_F)



# Determine scale and tail parameters for other distributions.
para['delta_H'] = ( para['alpha']*para['delta_F']*(1.0-1.0/para['sigma']) -
                    para['delta_F'] ) / ( (para['alpha']+para['delta_F'])*(1.0-1.0/para['sigma'])-1.0 )

para['lambda_Hw'] = para['lambda_F']/para['delta_F'] * para['delta_H'] /        \
                    np.dot(  z_grid**( (1.0/para['sigma']-1.0)/( para['alpha']*(1.0-1.0/para['sigma'])-1.0 )) , phi_F )


para['delta_G'] = para['delta_F'] * para['delta_H'] / (para['delta_H'] - para['delta_F'] - para['delta_H']*para['delta_F']) #






# ======================================================================================================================
#   Labor demand
# ======================================================================================================================
# Given the distribution of productivity, solve for the labor demand for a firm at each grid point
#
# $$ l(z) = N z^{\frac{ 1/\sigma -1 }{ \alpha(1-1/\sigma)-1 }}  \int_{0}^{\infty} x^{\frac{ 1/\sigma -1 }{ \alpha(1-1/\sigma)-1 }}   \phi_{F}(x)dz  $$
#
# and
#
# $$ \phi_{Hw}(z) = \frac{ z^{  \frac{1/\sigma-1}{\alpha(1-1/\sigma)-1 } }  \phi_{F}(z)}{  \int_{0}^{\infty} x^{  \frac{1/\sigma-1}{\alpha(1-1/\sigma)-1 } }  \phi_{F}(x) \, dx  }$$

def LaborDemand(para, z_grid, phi_F):


    # Labor demanded by a firm at each grid point
    #############################################
    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return  para['N'] * z_grid ** l_exp / np.dot( z_grid ** l_exp , phi_F )


l_z = LaborDemand(para, z_grid, phi_F)


# The distribution of labor demand across all firms
def DistWorkers(para, z_grid, phi_F):

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )
    return z_grid ** l_exp * phi_F / np.dot( z_grid ** l_exp , phi_F)

phi_Hw = DistWorkers(para, z_grid, phi_F)




# ======================================================================================================================
#   GDP/Output
# ======================================================================================================================
# $$ Y(t) =	 N^{\alpha}		  \left[ \int_{0}^{\infty} z^{  \frac{  1/\sigma -1  } { \alpha(1-1/\sigma)-1 }  }    	 \phi_{F}(z) \, dz \right]^{\sigma/(\sigma-1) - \alpha}	$$


def GDP(para, z_grid, phi_F):

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return para['N'] ** para['alpha'] * np.dot( z_grid ** l_exp , phi_F) ** ( para['sigma']/( para['sigma']-1.0) - para['alpha'])

Y = GDP(para, z_grid, phi_F)



# ======================================================================================================================
#   Wages
# ======================================================================================================================
# $$ w =  N^{\alpha(1-1/\sigma)-1}  Y(t)^{1/\sigma} \alpha(1-1/\sigma)   \left[ \int_{0}^{\infty}   x^{	\frac{1/\sigma-1 }{ \alpha(1-1/\sigma)-1 } } 		 \phi_{F}(x) \, dx	\right]^{ 1 -\alpha(1-1/\sigma)   } $$


def Wages(para, z_grid, phi_F):

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0) )

    return para['N'] ** ( para['alpha'] * ( 1.0 - 1/para['sigma'] )-1.0 ) * Y ** (1.0/para['sigma']) *    \
           para['alpha'] * (1.0-1.0/para['sigma'])  * np.dot( z_grid ** l_exp , phi_F) ** (1.0 - para['alpha'] * (1.0- 1.0/para['sigma']))

w = Wages(para, z_grid, phi_F)



# ======================================================================================================================
#   Firm Revenue
# ======================================================================================================================
# $$ p(z)y(z) =		N^{\alpha}	z^{	\frac{1/\sigma-1}{\alpha(1-1/\sigma)-1} }			\left[\int_{0}^{\infty}x^{\frac{1/\sigma-1}{\alpha(1-1/\sigma)-1}}\phi_{F}(x)\,dx\right]^{  \frac{1-\alpha \sigma + \alpha }{(\sigma-1)}
#  } $$


def FirmRev(para, z_grid, phi_F):

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return para['N'] ** para['alpha'] * z_grid ** l_exp * np.dot( z_grid ** l_exp , phi_F ) **      \
                                            ( ( 1.0 - para['alpha']*para['sigma']+para['alpha']) / ( para['sigma'] - 1.0 ) )


firm_rev = FirmRev(para, z_grid, phi_F)








# ======================================================================================================================
#   INITIAL GUESSES
# ======================================================================================================================
# Define the initial values for value functions and policy rules. This is needed to start the algorithm.



# Initial guesses for value functions
#----------------------------------------------------------

vf_Pi   = (firm_rev - w*l_z) / ( para['r'] - para['gamma'] ) # Value function of the firm

vf_W    = w / ( para['r'] - para['gamma'] ) * np.ones(grid_size)


vf_M = para['beta']/(1.0-para['beta'])*para['tau']*firm_rev + vf_W

vf_M[vf_M<vf_W] = vf_W[vf_M<vf_W] + 0.000001 # Being a manager is always at least as good as being a worker




# Initial guesses for policy rules
#----------------------------------------------------------

s_z = np.ones(grid_size)  # Guess for worker search intensity

v_z = np.ones(grid_size)  # Guess for firm's vacancy posting rate

phi_Hs = s_z  * phi_Hw / np.dot( s_z , phi_Hw  )


# Initial guess for labor market conditions
#----------------------------------------------------------

theta   = 1.0/para['N']       # Guess for labor market tightness
q_theta = para['q_norm'] * theta ** para['q_cd']    # Probability of matching with a worker given theta





#=======================================================================================================================
# FUNCTIONS AND SUBFUNCTIONS
#=======================================================================================================================


#=======================================================================================================================
# Knowledge transfer probability (g)
#=======================================================================================================================
def TransProbVec(indx_x, indx_y, para=para, z_grid=z_grid):
    """
    Transition Probability Vector
    Returns an array with the probability of the manager being able to transfer the amount of knowledge needed to reach any point between (and including) indx_x to indx_y


    Parameters
    ----------
    indx_x: Firm's level of productivity index

    indx_y: Potential manager's level of productivity index

    para: Dictionary of parameter values

    z_grid: model gridspace


    Returns
    -------
    Array: Probability that productivity z can be obtained

    """

    shape = 1.0/para['delta_G'] # Distribution's shape parameter

    # The take sub-sample of z_grid that is releveant (between indx_x and indx_y) and scale it relative to z_grid[indx_x]
    scaled_grid = z_grid[ indx_x:(indx_y+1) ] / z_grid[indx_x] # Have to add one on because Python doesnt include end


    # Compute PDF on each scalled grid point by computing the CDF at each point, and then taking the difference
    raw_CDF = np.hstack((1.0-scaled_grid ** (-shape), 1.0))

    return np.diff(raw_CDF)


def TransProbIndiv(indx_z, indx_x, indx_y, para=para, z_grid=z_grid):
    """
    Transition Probability Individual
    Returns the probability of transitioning to a single point indx_z.

    Calls on TransProbVec and then just gives you the single value you were looking for


    Parameters
    ----------
    indx_z: Position's productivity index

    indx_x: Firm's level of productivity index

    indx_y: Potential manager's level of productivity index

    para: Dictionary of parameter values

    z_grid: model gridspace

    Returns
    -------
    Float: Probability that productivity z can be obtained

    """

    prob_vec = TransProbVec(indx_x, indx_y, para=para, z_grid=z_grid)

    return prob_vec[indx_z-indx_x]




#=======================================================================================================================
# Matrix of moving
#=======================================================================================================================
# A move from $x$ to $z$ is only possible if the value of being a manager at $z'$ is higher than the value of being a worker for the other company ($y$).
#
# Here I compute a transition like matrix that shows whether a worker at $y$ wil accept a managerial job at $z$.

def CreateMoveMatrix(vf_M,vf_W):
    """
    Create a matrix of binary values showing which firms a worker will move to become a manager of given the value functions

    Parameters
    ----------
    vf_M: array_like(float)
       Value of being a manager at each grid point
    vf_W: array_like(float)
        Value of being a worker at each grid point

    Returns
    -------
    out_matrix: array
           Element (i,j) gives vf_M[z_grid[i]] > vf_W[z_grid[j]]

           So we can read row i as which managers will take a job at z_grid[i].
           And we can read column j as where will manager j take a job.
    """


    # Preallocate matrix
    out_matrix = np.zeros( (len(vf_M),len(vf_M)) )

    for indx_row in np.arange(0,len(vf_M)):

        out_matrix[indx_row,:] = vf_M[indx_row] > vf_W


    return out_matrix



#=======================================================================================================================
# Vacancy posting rate
#=======================================================================================================================
#
# $$ \nu(z) 	 = 	\frac{1}{\psi}  q(\tilde{\theta}^{[j]}) \int_{z}^{\infty} \left( \int_{\bar{x}(y)}^{y} \left[  \widetilde{\Pi}^{[j]} (x) - \widetilde{\Pi}^{[j]} (z) \right] g(x,z,y) \, dx \right) \, \phi_{Hs}^{[j]} (y)  \, dy  $$



def VacPost(indx_z, moving_matrix, para, q_theta, vf_Pi,  phi_Hs):
    """
    Compute the vacancy posting rate of a firm with productivity z_grid[indx_z]

    Parameters
    ----------
    indx_z: Index of the firm's productivity in z_grid

    para: Dictionary of model parameters

    q_theta: Probability of a firm matching with a worker

    vf_Pi: (Array) The value of being a firm at each grid point

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    phi_Hs: (Array) PDF of worker's search intensity distribution.


    Returns
    -------
    vacancy posint rate (float)
    """


    # Pre-allocate vector array to hold value of inner integral for each 'y'
    inner_intgr = np.empty(grid_size - indx_z, dtype=float)


    for indx_y in np.arange(indx_z,grid_size): # For each index of the array inner_intgr


        move_possible = moving_matrix[indx_z:(indx_y+1),indx_y-1] # will worker "y" accepting taking a job at x in [z,y]

        inner_intgr[indx_y-indx_z] =  np.dot( move_possible * (vf_Pi[indx_z:(indx_y+1)]-vf_Pi[indx_z]) , TransProbVec(indx_z, indx_y ) )


    return max([ 0 , q_theta/para['psi'] * np.dot( inner_intgr, phi_Hs[indx_z:]) ])



#=======================================================================================================================
# Search Threshold
#=======================================================================================================================
# The value at which Firms will stop firing people (because too many are leaving)
#
# $$ \varsigma < \frac{ \gamma z \iota^{'}(z)	}{ \iota(z) \theta q(\theta) \int_{0}^{z}[ \int_{\bar{x}(z)}^{z} g(x,y,z) dx ] \phi_{Fv}(y) \, dy  } $$

def s_thresh(indx_z, moving_matrix , phi_Fv , para, theta, q_theta):
    """
    Compute the search threshold for firm indx_z

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    para: Dictionary of model parameters

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker

    phi_Fv: Distribution of vacancies

    Returns
    -------
    s_threshold = (float) value at which firm will be indifferent between firing or not workers
    """

    s_thres_num = para['gamma']*( 1.0/para['sigma']-1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'] ) - 1.0 )



    inner_intgr = np.empty(indx_z+1, dtype=float) # Pre-allocate

    for indx_y in np.arange(0,indx_z+1):

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob , TransProbVec(indx_y,indx_z) )



    return s_thres_num / ( theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)]) )



#=======================================================================================================================
# Search intensity - case 2 (people not geting fired)
#=======================================================================================================================
# Computes the search intensity of workers at firm $z_n$ when none are being fired
#
# $$ 	c_{s}^{'}(s(z))		= \tilde{\theta} q({\theta}) \int_{0}^{z} \left( \int_{y}^{z} \mathbb{I}_{M(x)>W(z)}[\widetilde{M}(x)-\widetilde{W}(z)]g(x,y,z)  dx \right)  \phi_{Fv}(y)  dy		 $$


def SearchIntensity2(indx_z, moving_matrix, phi_Fv, para, vf_M, vf_W, theta, q_theta):
    """
    Compute the search intensity for workers at indx_z conditional upon no one is getting fired

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    para: Dictionary of model parameters

    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    phi_Fv: Distribution of vacancies

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker



    Returns
    -------
    s_z = (float) search intensity choice of a worker
    """


    inner_intgr = np.empty(indx_z+1, dtype=float) # Pre-allocate

    for indx_y in np.arange(0,indx_z+1):

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob * (vf_M[indx_y:(indx_z+1)] - vf_W[indx_z]), TransProbVec(indx_y,indx_z) )

    return 1.0/para['psi_s']* ( theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)]) )


#=======================================================================================================================
# Search intensity - case 1 (people are getting fired)
#=======================================================================================================================
#
# $$ 	c_{s}^{'}(\varsigma(z))		=	\tilde{\theta} q({\theta}) \int_{0}^{z} \left( \int_{y}^{z} \mathbb{I}_{M(x)>W(y)} [\widetilde{M}(x)-\widetilde{W}(z)] g(x,y,z) \, dx \right) \, \phi_{Fv}(y) \, dy   - \tilde{\theta} q(\tilde{\theta})  \left( \int_{0}^{z} [1-G(\bar{x}(z)] \phi_{Fv})(y) \, dy \right) \left(  \int_{0}^{\infty} ( \widetilde{W}(y) - \widetilde{W}(z)) \phi_{Fw}(y) \, dy 	\right)	 $$



def SearchIntensity1(indx_z, moving_matrix, phi_Fv, phi_Fw, para, vf_M, vf_W,  theta, q_theta):
    """
    Compute the search intensity for workers at indx_z conditional upon people being fired

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    para: Dictionary of model parameters

    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    phi_Fv: Distribution of vacancies

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker

    phi_Fw: PDF distribution of firms hiring workers


    Returns
    -------
    s_z = (float) search intensity choice of a worker
    """

    # First part of this equation is the same as case 2
    first_part = SearchIntensity2(indx_z, moving_matrix, para=para, vf_M=vf_M, vf_W=vf_W, phi_Fv=phi_Fv, theta=theta, q_theta=q_theta)


    # Second part

    Expected_vf_if_fired = np.dot( vf_W, phi_Fw) - vf_W[indx_z] # The expected value of being hired by a random hiring firm.


    inner_intgr = np.empty(indx_z+1)

    for indx_y in np.arange(0,indx_z+1):

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob, TransProbVec(indx_y,indx_z))

    Prob_worker_leaves = theta * q_theta * np.dot( inner_intgr, phi_Fv[0:(indx_z+1)])


    return first_part - 1.0/para['psi_s'] * Prob_worker_leaves * Expected_vf_if_fired




#=======================================================================================================================
#  Number of workers hired
#=======================================================================================================================
#
# $$ hire = 	\frac{ \widetilde{A}(z) + \widetilde{B}(z}{  \int_{0}^{\infty} \widetilde{A}(y) dy + \int_{0}^{\infty} \widetilde{B}(y) dy }  $$
#
#
# where
# $$	\widetilde{A}(z) 	=		\int_{0}^{z} \max \{ 0, \iota(z)-\iota(x)-1 \}  q(\tilde{\theta})\nu_{M}(x) \left[ \int_{\bar{x}(x')}^{\infty} g(z,x,x')  \phi_{Hs}(x')  dx' \right]  \phi_{F}(x)  dx	$$
#
# and
# $$ \widetilde{B}(z)	=		\max \left\{ 0  ,  \iota(z) \varsigma(y) \tilde{\theta}	q(\tilde{\theta}) \int_{0}^{z} \phi_{Fv}(x) [1-G(\bar{x},x,z)]  dx -\gamma z \iota^{'}(z)	\right\} \times	\max \left\{ 0  ,  \left( 1 - q(\tilde{\theta}) \nu_M(z) \int_{z}^{\infty} \phi_{Hs}(x^{'}) \int_{z}^{x^{'}} \mathbb{I}_{M(x)>W(x^{'})} g(x,z,x^{'})  dx  dx^{'} \right) \phi_{F}(z) \right\} $$

# <codecell>

def NumberWorkersHired(indx_z,moving_matrix, phi_Fv, grid_size, l_z, q_theta, v_z, phi_Hs, phi_F, s_z, theta,  para):
    """
    Compute the number of new hires by a firm at indx_z

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    phi_Fv: array of PDF for the distribution of vacancies being posted

    grid_size: how many grid points

    l_z: Labor demand across firms (array)

    q_theta: probability a firm hires a worker

    v_z: vacancy posting rate across state space grid

    phi_Hs: array of PDF of distribution of searching workers

    phi_Fs: array of PDF of distribution of firm productivity

    s_z: search intensity across grid space

    theta: labor market tightness

    para: Dictionary of model parameters


    Returns
    -------
    (float) Number of workers hired by average firm at z_grid
    """

    # PART A

    starting_prod = np.empty(indx_z+1)

    for indx_x in np.arange(0,indx_z+1): # between 0 and z

        g_vec = np.empty(grid_size-indx_z) # Pre-allocate vector

        for indx_xp in np.arange(indx_z,grid_size): # between z and infinity

            g_vec[indx_xp-indx_z] = TransProbIndiv(indx_z, indx_x, indx_xp)



        move_prob = moving_matrix[ indx_z , indx_z: ]
        starting_prod[indx_x] = max((0, l_z[indx_z] - l_z[indx_x] - 1.0  ) ) * q_theta * v_z[indx_x] * np.dot( move_prob * g_vec , phi_Hs[indx_z:] )


    A_z = np.dot(starting_prod , phi_F[0:(indx_z+1)])


    #########################################################################

    # PART B

    inner_intgr = np.empty(indx_z+1)

    for indx_x in np.arange(0,indx_z+1):

        inner_intgr[indx_x] = np.dot( moving_matrix[ indx_x:(indx_z+1) , indx_z] , TransProbVec(indx_x, indx_z) )



    B_z1 = l_z[indx_z] * s_z[indx_z] * theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)] ) - para['gamma'] * ( 1.0/para['sigma'] - 1.0 ) / ( para['alpha']*(1.0-1.0/para['sigma']) -1.0) * l_z[indx_z]





    inner_intgr = np.empty(grid_size-indx_z)

    for indx_xp in np.arange(indx_z,grid_size):

        inner_intgr[indx_xp-indx_z] = np.dot( moving_matrix[indx_z:(indx_xp+1),indx_xp] , TransProbVec(indx_z, indx_xp) )

    B_z2 = (1.0 - q_theta * v_z[indx_z] * np.dot( phi_Hs[indx_z:], inner_intgr )  ) * phi_F[indx_z]

    B_z = max( (0 , B_z1 ) ) * max( ( 0 , B_z2 ) )



    return A_z + B_z




#=======================================================================================================================
# New value function of worker
#=======================================================================================================================
#
# Holding the moving_matrix fixed, and the policy choices, the value function of a worker can be written in matrix form:
# $$ A W = b $$
#
# Where $A$ is a matrix, $W$=vf_W, and $b$ is a vector
#
#
# $$     r  \widetilde{W}(z_n) 	=		\omega + \gamma  \widetilde{W}(z_n) -  \gamma z_n \frac{\widetilde{W}(z_n)-\widetilde{W}z_{n-1}  }{dz}	    -  \tilde{c}_{s}(\varsigma(z_n)) 		 + \varsigma(z_n) \tilde{\theta} q(\tilde{\theta}) \sum_{x=z_1}^{z_n} [\frac{\beta}{1-\beta} [\tau p(x)y(x) ] ] \sum_{y=z_1}^{x}  \mathbb{I}_{M(x)>W(z)}  g(x,y,z_n) \, \phi_{Fv}(y) \, dy	 \, dx	+ \varsigma(z_n) \tilde{\theta} q(\tilde{\theta}) \sum_{x=z_1}^{z_n} [  \widetilde{W}(x) ] \sum_{y=z_1}^{x}  \mathbb{I}_{M(x)>W(z)}  g(x,y,z_n) \, \phi_{Fv}(y) \, dy	 \, dx	 - \varsigma(z_n) \tilde{\theta} q(\tilde{\theta}) \sum_{x=z_1}^{z_n}  \widetilde{W}(z)  \sum_{y=z_1}^{x}  \mathbb{I}_{M(x)>W(z)}  g(x,y,z_n) \, \phi_{Fv}(y) \, dy	 \, dx	+ \mathbb{I}_{\Delta l<0} \Bigg[   \frac{ \gamma z_n \iota^{'}(z_n ) }{ \iota(z_n) } - \varsigma(z_n) \tilde{\theta} q(\tilde{\theta})   \sum_{y=0}^{z} \sum_{x=y}^{z} \mathbb{I}_{M(x)>W(z)} g(x,y,z) dx  \phi_{Fv}(y) dy \Bigg]  \times     \sum_{y'=0}^{\infty}  (\widetilde{W}(y')-\widetilde{W}(z))  \phi_{Fw}(y') \, d y'		 +    q(\tilde{\theta}) \nu_{M}(z) \sum_{x=z_n}^{\infty} (\widetilde{W}(x)-\widetilde{W}(z))  \sum_{y=x}^{\infty} \mathbb{I}_{M(x)>W(y)}   g(x,z,y) \, \phi_{Hs}(y)\, dy  \, dx           $$


def GetNewVfW(moving_matrix, para, grid_size, dz, phi_Fv, firm_rev, theta, q_theta, s_z, v_z, phi_Fw, phi_Hs):


    # Pre-allocate matrices

    A = np.zeros( (grid_size , grid_size) )

    b = np.zeros( grid_size )

    # Start filling in points for each grid point

    for indx_z in np.arange(0,grid_size):


        b[indx_z] = -w + para['psi_s']/2 * s_z[indx_z] ** 2

        A[indx_z,indx_z] = A[indx_z,indx_z] - para['r'] + para['gamma']


        if indx_z > 0:

            A[indx_z,indx_z] = A[indx_z,indx_z]  - para['gamma'] * z_grid[indx_z] / dz

            A[indx_z,(indx_z-1)] = A[indx_z,(indx_z-1)] + para['gamma'] * z_grid[indx_z] / dz



        for indx_x in np.arange(0,indx_z+1):


            inner_intgr = np.empty(indx_x+1)

            for indx_y in np.arange(0,indx_x+1):

                inner_intgr[indx_y] = moving_matrix[indx_x,indx_z] * TransProbIndiv( indx_x, indx_y, indx_z) * phi_Fv[indx_y]


            b[indx_z] = b[indx_z] - s_z[indx_z] * theta * q_theta * para['beta'] / ( 1.0 - para['beta'] ) * para['tau'] * firm_rev[indx_x] *  sum(inner_intgr)

            A[indx_z, indx_x] = A[indx_z, indx_x] + s_z[indx_z] * theta * q_theta * sum(inner_intgr)

            A[indx_z, indx_z] = A[indx_z, indx_z] - s_z[indx_z] * theta * q_theta * sum(inner_intgr)



        for indx_y in np.arange(0,grid_size):


            A[indx_z,indx_y] =  A[indx_z,indx_y] + ProbOfFired(indx_z,moving_matrix,para=para, theta=theta, q_theta=q_theta, phi_Fv=phi_Fv,s_z=s_z) * phi_Fw[indx_y]

            A[indx_z,indx_z] =  A[indx_z,indx_z] - ProbOfFired(indx_z,moving_matrix,para=para, theta=theta, q_theta=q_theta, phi_Fv=phi_Fv,s_z=s_z) * phi_Fw[indx_y]




        for indx_x in np.arange(indx_z,grid_size):

            g_vec = np.empty(grid_size-indx_x)

            for indx_y in np.arange(indx_x,grid_size):
                g_vec[indx_y-indx_x] = TransProbIndiv(indx_x,indx_z,indx_y)


            A[indx_z,indx_x] = A[indx_z,indx_x] + q_theta * v_z[indx_z] * np.dot(  moving_matrix[indx_x, indx_x: ] * g_vec , phi_Hs[indx_x:]  )

            A[indx_z,indx_z] = A[indx_z,indx_z] - q_theta * v_z[indx_z] * np.dot(  moving_matrix[indx_x, indx_x: ] * g_vec , phi_Hs[indx_x:]  )



    # Solve model
    tmp = np.linalg.solve(A, b)

    return tmp



#=======================================================================================================================
# Probability of worker being fired
#=======================================================================================================================
#
# $$ \mathbb{I}_{\Delta l<0} \Bigg[   \frac{ \gamma z_n \iota^{'}(z_n ) }{ \iota(z_n) } - \varsigma(z_n) \tilde{\theta} q(\tilde{\theta})   \sum_{y=0}^{z} \sum_{x=y}^{z} \mathbb{I}_{M(x)>W(z)} g(x,y,z) dx  \phi_{Fv}(y) dy \Bigg] $$

def ProbOfFired(indx_z,moving_matrix,para, theta, q_theta, phi_Fv,s_z):


    inner_intgr = np.empty(indx_z+1 )

    for indx_y in np.arange(0,(indx_z+1) ):

        inner_intgr[indx_y] =  np.dot( moving_matrix[ indx_y:(indx_z+1) , indx_z] , TransProbVec(indx_y,indx_z) )

    # Desired (abs) change in labor - num. workers who leave
    tmp = para['gamma'] * ( 1.0/para['sigma'] - 1.0 ) / ( para['alpha']*(1.0-1.0/para['sigma']) - 1.0 )    - s_z[indx_z] * theta * q_theta * np.dot(inner_intgr , phi_Fv[:(indx_z+1) ] )

    return max( (0,tmp) )




#=======================================================================================================================
#   New value function of the firm
#=======================================================================================================================
#
# Holding the moving_matrix fixed, and the policy choices, and the managerial payment ($\mu$), the value function of a firm can be written in matrix form:
# $$ A \Pi = b $$
#
# Where $A$ is a matrix, $\Pi$=vf_Pi, and $b$ is a vector

def GetNewVfPi(mu_z, moving_matrix, para, firm_rev, w, l_z, v_z, dz, q_theta ):


   # Pre-allocate matrices

    A = np.zeros( (grid_size , grid_size) )

    b = np.zeros( grid_size )

    # Start filling in points for each grid point

    for indx_z in np.arange(0,grid_size):


        b[indx_z] = mu_z[indx_z] - firm_rev[indx_z] + w*l_z[indx_z] + para['psi']/2 * v_z[indx_z] ** 2


        A[indx_z,indx_z] = A[indx_z,indx_z] + para['gamma'] - para['r']



        if indx_z > 0:

            A[indx_z,indx_z] = A[indx_z,indx_z]  - para['gamma'] * z_grid[indx_z] / dz

            A[indx_z,(indx_z-1)] = A[indx_z,(indx_z-1)] + para['gamma'] * z_grid[indx_z] / dz



        inner_intgr = np.empty(grid_size-indx_z)

        for indx_x in np.arange(indx_z,grid_size):


            g_vec = np.empty(grid_size-indx_x)

            for indx_y in np.arange(indx_x,grid_size):
                g_vec[indx_y-indx_x] = TransProbIndiv(indx_x,indx_z,indx_y)



            A[indx_z,indx_x] = A[indx_z,indx_x] + q_theta * v_z[indx_z] * np.dot( moving_matrix[indx_x, indx_x:] * g_vec  , phi_Hs[indx_x:] )

            A[indx_z,indx_z] = A[indx_z,indx_z] - q_theta * v_z[indx_z] * np.dot( moving_matrix[indx_x, indx_x:] * g_vec  , phi_Hs[indx_x:] )



    # Solve system of equations
    tmp = np.linalg.solve(A, b)

    return tmp



#=======================================================================================================================
#   Solve for the policy rules (given phi_F and value functions).
#=======================================================================================================================
#


def FindPolicyRules(v_z, s_z, vf_M, vf_W, vf_Pi, para, q_theta, phi_Hs, phi_F, phi_Hw, l_z):


    moving_matrix = CreateMoveMatrix(vf_M, vf_W)


    for iter_count_pol in np.arange(0,max_pol_iter+1):

        v_z_old = np.copy(v_z)
        s_z_old = np.copy(s_z)

        for indx_zn in np.arange(0,grid_size): # For each grid point

            # Update vacancy posting rate of firm
            #============================================================
            v_z_new = VacPost(indx_zn, moving_matrix, para=para, q_theta=q_theta, vf_Pi=vf_Pi,  phi_Hs=phi_Hs) # new vacancy posting rate for firm indx_zn

            v_z[indx_zn] = ff_pol * v_z_new + (1.0-ff_pol) * v_z_old[indx_zn] # Applying the fudge factor


            # Update distribution of vacancy postings
            #============================================================
            phi_Fv = v_z * phi_F / np.dot( v_z , phi_F  )


            # Update labor market tightness
            #============================================================
            theta = np.dot( v_z , phi_F ) / np.dot( para['N'] * s_z , phi_Hw)

            q_theta = para['q_norm'] * theta ** para['q_cd']


            #Update distribution of worker hires
            #============================================================

            new_hires = np.empty(grid_size)

            for indx_zj in np.arange(0,grid_size): # For each firm on the grid, computer the number of new hires

                new_hires[indx_zj] = NumberWorkersHired(indx_zj,moving_matrix,phi_Fv, grid_size=grid_size, l_z=l_z, q_theta=q_theta, v_z=v_z, phi_Hs=phi_Hs, phi_F=phi_F, s_z=s_z, theta=theta,  para=para)


            phi_Fw = new_hires / sum(new_hires) # Convert number of hires into PDF



            # Update Worker's search intensity
            #============================================================

            # Compute threshold
            search_threshold = s_thresh(indx_zn, moving_matrix , phi_Fv , para=para, theta=theta, q_theta=q_theta)

            search1 = SearchIntensity1(indx_zn, moving_matrix, phi_Fv, phi_Fw, para=para, vf_M=vf_M, vf_W=vf_W,  theta=theta, q_theta=q_theta)
            search2 = SearchIntensity2(indx_zn, moving_matrix, phi_Fv, para=para, vf_M=vf_M, vf_W=vf_W, theta=theta, q_theta=q_theta)

            s_z_new = (search1<search_threshold) * search1 + (search2>search_threshold)*search2




            s_z[indx_zn] = ff_pol*max([s_z_new, 0]) + (1.0-ff_pol)*s_z_old[indx_zn]


            phi_Hs = s_z * phi_Hw  / np.dot( s_z , phi_Hw )



            # Update labor market tightness
            #============================================================
            theta = np.dot( v_z , phi_F ) / np.dot( para['N'] * s_z , phi_Hw)

            q_theta = para['q_norm'] * theta ** para['q_cd']



        s_diff = max(abs(s_z - s_z_old))
        v_diff = max(abs(v_z - v_z_old))

        max_pol_diff = max([s_diff, v_diff])


        if max_pol_diff < tol_pol:
            break



    return v_z, s_z, iter_count_pol













########################################################################################################################
#                                        EXECUTE THE ALGORITHM
########################################################################################################################


phi_F_old = np.copy(phi_F)
gamma_old = para['gamma']

#=======================================================================================================================
# Step zero: Given phi_F, update variable that only depend on it
#=======================================================================================================================
l_z         = LaborDemand(para, z_grid, phi_F)
phi_Hw      = DistWorkers(para, z_grid, phi_F)
#Y           = GDP(para, z_grid, phi_F)
w           = Wages(para, z_grid, phi_F)
firm_rev    = FirmRev(para, z_grid, phi_F)



#=======================================================================================================================
# Step 1) Given phi_F, solve for value function and policy rules
#=======================================================================================================================

for iter_count_vf in np.arange(1,max_vf_iter+1):

    print iter_count_vf

    vf_M_old = np.copy(vf_M)
    vf_W_old = np.copy(vf_W)
    vf_Pi_old = np.copy(vf_Pi)



    #-------------------------------------------------------------------------------------------------------------------
    # Step 1.1) Given phi_F and value functions, find policy rule
    #-------------------------------------------------------------------------------------------------------------------
    v_z, s_z, iter_count_pol = FindPolicyRules(v_z, s_z, vf_M, vf_W, vf_Pi, para, q_theta, phi_Hs, phi_F, phi_Hw, l_z)

    if iter_count_pol == max_pol_iter:
        print "Warning: maximum number of policy iterations exceeded. Results may be inacuurate"




    #---------------------------------------------------------------------------------------------------------------
    # Step 1.2) Given phi_F and policy rules, update value functions
    #---------------------------------------------------------------------------------------------------------------
    moving_matrix = CreateMoveMatrix(vf_M,vf_W)

    phi_Fv = v_z * phi_F / np.dot( v_z , phi_F  )
    theta = np.dot( v_z , phi_F ) / np.dot( para['N'] * s_z , phi_Hw)
    q_theta = para['q_norm'] * theta ** para['q_cd']
    phi_Hs  =   s_z * phi_Hw  / np.dot( s_z , phi_Hw )

    new_hires = np.empty(grid_size)
    for indx_zj in np.arange(0,grid_size): # For each firm on the grid, computer the number of new hires
            new_hires[indx_zj] = NumberWorkersHired(indx_zj,moving_matrix,phi_Fv, grid_size=grid_size, l_z=l_z, q_theta=q_theta, v_z=v_z, phi_Hs=phi_Hs, phi_F=phi_F, s_z=s_z, theta=theta,  para=para)
    phi_Fw = new_hires / sum(new_hires) # Convert number of hires into PDF



    # Make new draw for vf_W
    vf_W_new = GetNewVfW(moving_matrix, para=para, grid_size=grid_size, dz=dz, phi_Fv=phi_Fv, firm_rev=firm_rev, theta=theta, q_theta=q_theta, s_z=s_z, v_z=v_z, phi_Fw=phi_Fw, phi_Hs=phi_Hs)

    # Compute implied values for vf_M
    vf_M_new = para['beta']/(1.0-para['beta'])*para['tau']*firm_rev + vf_W_new



    moving_matrix_new = CreateMoveMatrix(vf_M_new,vf_W_new)


    # Given vf_M, solve for the managerial payment mu
    mu_z = np.empty(grid_size)

    for indx_z in np.arange(0,grid_size):
        if indx_z>0:
            dMdz = ( vf_M_new[indx_z] - vf_M_new[indx_z-1] ) / dz
        else:
            dMdz = 0

        inner_intgr = np.empty( grid_size - indx_z )

        for indx_y in np.arange(indx_z , grid_size ):

            inner_intgr[indx_y-indx_z] =  np.dot( moving_matrix_new[ indx_z:(indx_y+1) ,indx_y ] * ( vf_W_new[ indx_z:(indx_y+1) ] - vf_M_new[indx_z] )  , TransProbVec(indx_z, indx_y) )


        mu_z[indx_z] = ( para['r'] - para['gamma'] )*vf_M_new[indx_z] + para['gamma']*z_grid[indx_z] * dMdz - q_theta * v_z[indx_z] * np.dot( inner_intgr , phi_Hs[indx_z:] )




    # Solve for the firm's value function
    vf_Pi_new = GetNewVfPi(mu_z, moving_matrix_new, para, firm_rev, w, l_z, v_z, dz, q_theta )


    # Check for convergence of the value function
    max_Pi_diff = max(abs(vf_Pi_new-vf_Pi_old))
    max_W_diff  = max(abs(vf_W_new-vf_W_old))
    max_M_diff  = max(abs(vf_M_new-vf_M_old))


    max_vf_diff = max([max_M_diff, max_W_diff, max_Pi_diff ])


    if max_vf_diff < tol_vf:
        break
    else:
        vf_Pi = ff_pol*vf_Pi_new + (1.0-ff_pol)*vf_Pi_old
        vf_M  = ff_pol*vf_M_new  + (1.0-ff_pol)*vf_M_old
        vf_W  = ff_pol*vf_W_new  + (1.0-ff_pol)*vf_W_old






print "Model has completed running"