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




# ======================================================================================================================
# PARAMETERIZATION OF THE MODEL
# ======================================================================================================================
# Here the user can input the parameterization of the model. Aside from the settings in this cell, no other
# user input is required in the rest of the code.



# Define the state-space grid of the model
#----------------------------------------------------------
# The continuous state space of productivity is approximated by a discrete grid

grid_size = 25        # Number of grid points
grid_max  = 5.0       # Maximum value in the grid
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

# Fudge factor (ff) = how much weight to give to the new draw (and 1-ff to the old draw) when updating the value of
# variables each iteration. Note, a lower "ff" value means the code takes longer to run, but is generally more stable in
# converging.
ff_pol      = 0.1   # Fudge factor when updating policy rules
ff_vf       = 0.1   # Fudge factor when updating value functions
ff_phi      = 0.1   # Fudge factor when updating the productivity distribution and growth rate

# Convergence tolerance criteria = determines when to stop iterating because convergence has been reached.
tol_pol     = 0.0001    # How accurate convergence in policy rules must be
tol_vf      = 0.0001    # How accurate convergence in value functions must be
tol_phi     = 0.0001    # How accurate convergence in distribution must be


# Maximum number of iterations to try
max_pol_iter    = 300   # Maximum number of iterations to try before we declare policy code does not converge
max_vf_iter     = 300   # Maximum number of iterations to try before we declare value function code does not converge
max_phi_iter    = 300   # Maximum number of iterations to try before we declare productivity dist code does not converge
max_main_iter   = 300   # Maximum number of iterations to run in the main loop

print_skip = 10 # Print the convergence progress for every `print_skip'-th iteration






# ======================================================================================================================
#   OTHER PARAMETERS IN THE MODEL
# ======================================================================================================================
# Based on the user input above, define other model parameters and distributions that are a function of the parameters
# already defined.


# Construct the initial guess for the firm productivity distribution based on the Frechet distribution
phi_F =  para['lambda_F'] / para['delta_F'] * z_grid ** ( -1.0/para['delta_F'] - 1.0 )          \
         * np.exp( -para['lambda_F'] * z_grid ** ( - 1/para['delta_F'] ) )


# Renormalize scale parameter
para['lambda_F'] = para['lambda_F'] / sum(phi_F)

phi_F = phi_F / sum(phi_F) # Now the sum of the PDF is one.


# Determine scale and tail parameters for other distributions.
#----------------------------------------------------------

# Tail parameter - dist of workers
para['delta_Hw'] = ( para['alpha']*para['delta_F']*(1.0-1.0/para['sigma']) -
                    para['delta_F'] ) / ( (para['alpha']+para['delta_F'])*(1.0-1.0/para['sigma'])-1.0 )

# Tail parameter - knowledge transfer distribution
para['delta_G'] = para['delta_F'] * para['delta_Hw'] / (para['delta_Hw'] - para['delta_F'] - para['delta_Hw']*para['delta_F'])







# ======================================================================================================================
#  DEFINE FUNCTIONS
# ======================================================================================================================
# Define the functions used to construct various distributions and variables throughout the model. Due to the iterative
# nature of the model's algorithm, we will have to construct these variable many times. So defining them as functions
# rather than constructing them in-line makes the code more compact and easier to read



def labor_demand(para, z_grid, phi_F):
    """
    Compute the array of labor demand by a firm at each point on the state grid.

    Parameters
    ----------
     para: Dictionary of parameter values

    z_grid: model productivity grid space

    phi_F: PDF of firm productivity over the grid space


    Returns
    -------
    Array: Number of workers demanded by a firm at each point on the grid space

    """

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return  para['N'] * z_grid ** l_exp / np.dot( z_grid ** l_exp , phi_F )


def dist_workers(para, z_grid, phi_F):
    """
    Compute the distribution of workers over the state grid.

    Parameters
    ----------
    para: Dictionary of parameter values

    z_grid: model productivity grid space

    phi_F: PDF of firm productivity over the grid space


    Returns
    -------
    Array: PDF of the labor distribution over the state grid
    """

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return z_grid ** l_exp * phi_F / np.dot( z_grid ** l_exp , phi_F)


def gdp(para, z_grid, phi_F):
    """
    Compute Aggregate output (gdp) in the economy.

    Parameters
    ----------
    para: Dictionary of parameter values

    z_grid: model productivity grid space

    phi_F: PDF of firm productivity over the grid space


    Returns
    -------
    Float: Real output in the economy
    """

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return para['N'] ** para['alpha'] * np.dot( z_grid ** l_exp , phi_F) ** ( para['sigma']/( para['sigma']-1.0) - para['alpha'])


def wages(para, z_grid, phi_F):
    """
    Compute the market clearing wage rate for workers in the economy.

    Parameters
    ----------
    para: Dictionary of parameter values

    z_grid: model productivity grid space

    phi_F: PDF of firm productivity over the grid space


    Returns
    -------
    Float: Worker's wage rate in the economy

    """

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0) )

    Y = gdp(para, z_grid, phi_F)

    return para['N'] ** ( para['alpha'] * ( 1.0 - 1/para['sigma'] )-1.0 ) * Y ** (1.0/para['sigma']) *    \
           para['alpha'] * (1.0-1.0/para['sigma'])  * np.dot( z_grid ** l_exp , phi_F) ** (1.0 - para['alpha'] * (1.0- 1.0/para['sigma']))


def firm_revenue(para, z_grid, phi_F):
    """
    Compute the array of firm revenue at each point on the state grid.

    Parameters
    ----------
    para: Dictionary of parameter values

    z_grid: model productivity grid space

    phi_F: PDF of firm productivity over the grid space


    Returns
    -------
    Array: Firm revenue at each point on the grid space
    """

    l_exp = ( ( 1.0/para['sigma'] -1.0 ) / ( para['alpha']*( 1.0-1.0/para['sigma'])-1.0 ) )

    return para['N'] ** para['alpha'] * z_grid ** l_exp * np.dot( z_grid ** l_exp , phi_F ) **      \
                                            ( ( 1.0 - para['alpha']*para['sigma']+para['alpha']) / ( para['sigma'] - 1.0 ) )


def trans_prob_vec(indx_x, indx_y, para=para, z_grid=z_grid):
    """
    Transition Probability Vector
    Returns an array with the probability of the manager being able to transfer the amount of knowledge needed to reach
    each point between (and including) indx_x to indx_y


    Parameters
    ----------
    indx_x: Firm's level of productivity index

    indx_y: Potential manager's level of productivity index

    para: Dictionary of parameter values

    z_grid: model grid space


    Returns
    -------
    Array: Probability that productivity z can be obtained
    """

    shape = 1.0/para['delta_G'] # Distribution's shape parameter

    # The take sub-sample of z_grid that is relevant (between indx_x and indx_y) and scale it relative to z_grid[indx_x]
    scaled_grid = z_grid[ indx_x:(indx_y+1) ] / z_grid[indx_x] # Have to add one on because Python doesnt include end element


    # Compute PDF on each scaled grid point by computing the CDF at each point, and then taking the difference
    raw_CDF = np.hstack((1.0-scaled_grid ** (-shape), 1.0)) # Add 1 to the end of the array to give the value at final point

    return np.diff(raw_CDF)


def trans_prob_indiv(indx_z, indx_x, indx_y, para=para, z_grid=z_grid):
    """
    Transition Probability Individual
    Returns the probability of transitioning to a single point indx_z in [indx_x, indx_y]

    Calls trans_prob_vec and then just gives you the single value you were looking for


    Parameters
    ----------
    indx_z: Position's productivity index

    indx_x: Firm's level of productivity index

    indx_y: Potential manager's level of productivity index

    para: Dictionary of parameter values

    z_grid: model grid space

    Returns
    -------
    Float: Probability that productivity z can be obtained
    """

    prob_vec = trans_prob_vec(indx_x, indx_y, para, z_grid)

    return prob_vec[indx_z-indx_x]


def create_move_matrix(vf_M,vf_W):
    """
    Create a matrix of binary values showing which firms a worker will move to become a manager of given the value
    functions.

    The way to read the matrix is as follows:
    *   Column j shows whether of not a worker at a firm on the j-th state grid value searching for a managerial job will
            accept a managerial job at a firm if they were able to impart knowledge level i on the firm

    *   Row i, shows which search workers will accept a managerial job at a firm that has productivity level z_grid[i]
            after the worker takes the job

    Parameters
    ----------
    vf_M: array_like(float)
       Value of being a manager at each grid point
    vf_W: array_like(float)
        Value of being a worker at each grid point

    Returns
    -------
    out_matrix: array (grid_size x grid_size)
           Element (i,j) gives vf_M[z_grid[i]] > vf_W[z_grid[j]]
    """

    # Preallocate matrix
    out_matrix = np.zeros( (len(vf_M),len(vf_M)) )

    for indx_row in np.arange(0,len(vf_M)):

        out_matrix[indx_row,:] = vf_M[indx_row] > vf_W

    return out_matrix


def vac_post_rate(indx_z, moving_matrix, para, q_theta, vf_Pi,  phi_Hs):
    """
    Compute the vacancy posting rate of a firm with productivity z_grid[indx_z]

    Parameters
    ----------
    indx_z: Index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states are possible (which manager jobs workers will accept)

    para: Dictionary of model parameters

    q_theta: Probability of a firm matching with a worker

    vf_Pi: (Array) The value of being a firm at each grid point

    phi_Hs: (Array) PDF of worker's search intensity distribution.


    Returns
    -------
    Float: vacancy posing rate of firm with productivity z_grid[indx_z]
    """


    # Pre-allocate vector array to hold value of inner integral for each 'indx_y' between indx_z and the end
    inner_intgr = np.empty(grid_size - indx_z, dtype=float)


    for indx_y in np.arange(indx_z,grid_size): # For each index of the array inner_intgr

        move_possible = moving_matrix[indx_z:(indx_y+1),indx_y-1] # will worker "y" accepting taking a job at x in [z,y]

        inner_intgr[indx_y-indx_z] =  np.dot( move_possible * ( vf_Pi[indx_z:(indx_y+1)]-vf_Pi[indx_z] ) , trans_prob_vec(indx_z, indx_y ) )


    return max([ 0 , q_theta/para['psi'] * np.dot( inner_intgr, phi_Hs[indx_z:]) ]) # Vacancy posting rate cannot be negative


def s_thresh(indx_z, moving_matrix , phi_Fv , para, theta, q_theta):
    """
    Compute the search threshold for firm indx_z. When workers at firm indx_z search with an intensity above the
    threshold, enough of them leave to become managers that the firm does not want to fire anyone (but will instead hire).
    When the workers search with an intensity below the threshold, the firm fires workers to adjust the firm to the
    optimal labor size each period.

    This threshold value is important in determining the search intensity choice of workers.

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    phi_Fv: Distribution of managerial job vacancies

    para: Dictionary of model parameters

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker


    Returns
    -------
    s_threshold = (float) value at which firm will be indifferent between firing works and not
    """

    # Numerator of the threshold equation:
    s_thres_num = para['gamma'] * ( 1.0 / para['sigma'] - 1.0 ) / ( para['alpha'] * ( 1.0 - 1.0 / para['sigma'] ) - 1.0 )


    # Construct the denominator:
    inner_intgr = np.empty(indx_z+1, dtype=float) # Pre-allocate

    for indx_y in np.arange(0,indx_z+1): # For each grid point below indx_z

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob , trans_prob_vec(indx_y,indx_z) )

    # inner_intgr is the expected payoff from moving to firm indx_y

    return s_thres_num / ( theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)] ) )


def search_intensity2(indx_z, moving_matrix, phi_Fv, para, vf_M, vf_W, theta, q_theta):
    """
    Compute the search intensity for workers at indx_z conditional upon the firm NOT firing

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    phi_Fv: Distribution of vacancies

    para: Dictionary of model parameters

    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker


    Returns
    -------
    s_z = (float) search intensity choice of a worker at the firm indx_z
    """


    inner_intgr = np.empty(indx_z+1, dtype=float) # Pre-allocate

    for indx_y in np.arange(0,indx_z+1):

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob * (vf_M[indx_y:(indx_z+1)] - vf_W[indx_z]), trans_prob_vec(indx_y,indx_z) )

    return 1.0/para['psi_s']* ( theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)]) )


def search_intensity1(indx_z, moving_matrix, phi_Fv, phi_Fw, para, vf_M, vf_W,  theta, q_theta):
    """
    Compute the search intensity for workers at indx_z conditional upon the firm firing

    Parameters
    ----------
    indx_z: index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states is possible (which workers will accept)

    phi_Fv: Distribution of vacancies

    para: Dictionary of model parameters

    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    theta: labor market tightness

    q_theta: probability a firm is matched with a worker


    Returns
    -------
    s_z = (float) search intensity choice of a worker
    """

    # First part of this equation is the same as case 2
    first_part = search_intensity2(indx_z, moving_matrix, para=para, vf_M=vf_M, vf_W=vf_W, phi_Fv=phi_Fv, theta=theta, q_theta=q_theta)


    # Second part (additional term account for how search intensity impacts upon probability of being fired)

    Expected_vf_if_fired = np.dot( vf_W, phi_Fw) - vf_W[indx_z] # The expected value of being hired by a random hiring firm.


    inner_intgr = np.empty(indx_z+1)

    for indx_y in np.arange(0,indx_z+1):

        move_prob = moving_matrix[indx_y:(indx_z+1),indx_z]

        inner_intgr[indx_y] = np.dot( move_prob, trans_prob_vec(indx_y,indx_z))

    Prob_worker_leaves = theta * q_theta * np.dot( inner_intgr, phi_Fv[0:(indx_z+1)])


    return first_part - 1.0/para['psi_s'] * Prob_worker_leaves * Expected_vf_if_fired


def number_workers_hired(indx_z, moving_matrix, phi_Fv, grid_size, l_z, q_theta, v_z, phi_Hs, phi_F, s_z, theta,  para):
    """
    Compute the number of new hires by a firm at indx_z.

    The number of new hires at indx_z can be decomposed into two parts:
    *   Part A: firms who meet a manager and improve their productivity to indx_z and want to expand their firm size as
            a result

    *   Part B: Firms who were originally at indx_z, didnt improve their productivity, but had "lots" of workers leave
            to become managers elsewhere, and wants to replace (some of) them.

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

    starting_prod = np.empty(indx_z+1) # Preallocate array

    for indx_x in np.arange(0,indx_z+1): # between 0 and z

        g_vec = np.empty(grid_size-indx_z) # Pre-allocate vector

        for indx_xp in np.arange(indx_z,grid_size): # between z and infinity

            g_vec[indx_xp-indx_z] = trans_prob_indiv(indx_z, indx_x, indx_xp)


        move_prob = moving_matrix[ indx_z , indx_z: ]

        starting_prod[indx_x] = max((0.0 , l_z[indx_z] - l_z[indx_x] - 1.0  ) ) * q_theta * v_z[indx_x] * np.dot( move_prob * g_vec , phi_Hs[indx_z:] )


    A_z = np.dot(starting_prod , phi_F[0:(indx_z+1)])


    #########################################################################

    # PART B

    inner_intgr = np.empty(indx_z+1) # Preallocate array

    for indx_x in np.arange(0,indx_z+1):

        inner_intgr[indx_x] = np.dot( moving_matrix[ indx_x:(indx_z+1) , indx_z] , trans_prob_vec(indx_x, indx_z) )


    B_z1 = l_z[indx_z] * s_z[indx_z] * theta * q_theta * np.dot( inner_intgr , phi_Fv[0:(indx_z+1)] ) -     \
           para['gamma'] * ( 1.0/para['sigma'] - 1.0 ) / ( para['alpha']*(1.0-1.0/para['sigma']) -1.0) * l_z[indx_z]



    inner_intgr = np.empty(grid_size-indx_z) # Preallocate array

    for indx_xp in np.arange(indx_z,grid_size):

        inner_intgr[indx_xp-indx_z] = np.dot( moving_matrix[indx_z:(indx_xp+1),indx_xp] , trans_prob_vec(indx_z, indx_xp) )

    B_z2 = (1.0 - q_theta * v_z[indx_z] * np.dot( phi_Hs[indx_z:], inner_intgr )  ) * phi_F[indx_z]

    B_z = max( (0.0 , B_z1 ) ) * max( ( 0.0 , B_z2 ) )


    return A_z + B_z


def get_new_vf_W( moving_matrix, para, grid_size, dz, phi_Fv, firm_rev, theta, q_theta, s_z, v_z, phi_Fw, phi_Hs ):
    """
    Given the policy rules (and holding the moving_matrix constant), compute the new value of being a worker at each
    grid point.

    We can write the worker's value function in matrix notation: A W = b

    Where W is an array of the value function of workers at each grid point, A is a (grid_size x grid_size) matrix,
    and b is an array.

    See documentation for exact equations being constructed here

    Parameters
    ----------
    moving_matrix: Matrix showing which transitions between states are possible (which manager jobs workers will accept)

    para: Dictionary of model

    grid_size: How many points in out state space grid

    dz: step size between each productivity level in the grid space

    phi_Fv: PDF of managerial job vacancy postings

    firm_rev: revenue of a firm at each grid point

    theta: labor market tightness

    q_theta: Probability of a firm matching with a worker

    s_z: policy rule worker search intensity

    v_z: policy rule vacancy posting rate

    phi_Fw: PDF of new worker hires

    phi_Hs: PDF of worker's search intensity distribution.


    Returns
    -------
    Array: Value of being a worker at each grid point
    """

    # Pre-allocate matrices to fill

    A = np.zeros( (grid_size , grid_size) )

    b = np.zeros( grid_size )


    # Start filling in points for each grid point
    for indx_z in np.arange(0,grid_size):


        b[indx_z] = -w + para['psi_s']/2 * s_z[indx_z] ** 2

        A[indx_z,indx_z] = - para['r'] + para['gamma']


        if indx_z > 0:

            A[indx_z,indx_z] = A[indx_z,indx_z]  - para['gamma'] * z_grid[indx_z] / dz

            A[indx_z,(indx_z-1)] = A[indx_z,(indx_z-1)] + para['gamma'] * z_grid[indx_z] / dz



        for indx_x in np.arange(0,indx_z+1):


            inner_intgr = np.empty(indx_x+1)

            for indx_y in np.arange(0,indx_x+1):

                inner_intgr[indx_y] = moving_matrix[indx_x,indx_z] * trans_prob_indiv( indx_x, indx_y, indx_z) * phi_Fv[indx_y]


            b[indx_z] = b[indx_z] - s_z[indx_z] * theta * q_theta * para['beta'] / ( 1.0 - para['beta'] ) * para['tau'] * firm_rev[indx_x] *  sum(inner_intgr)

            A[indx_z, indx_x] = A[indx_z, indx_x] + s_z[indx_z] * theta * q_theta * sum(inner_intgr)

            A[indx_z, indx_z] = A[indx_z, indx_z] - s_z[indx_z] * theta * q_theta * sum(inner_intgr)



        for indx_y in np.arange(0,grid_size):


            A[indx_z,indx_y] =  A[indx_z,indx_y] + prob_being_fired(indx_z, moving_matrix, para,  theta,  q_theta,  phi_Fv, s_z) * phi_Fw[indx_y]

            A[indx_z,indx_z] =  A[indx_z,indx_z] - prob_being_fired(indx_z, moving_matrix, para,  theta,  q_theta,  phi_Fv, s_z) * phi_Fw[indx_y]



        for indx_x in np.arange(indx_z,grid_size):

            g_vec = np.empty(grid_size-indx_x)

            for indx_y in np.arange(indx_x,grid_size):
                g_vec[indx_y-indx_x] = trans_prob_indiv(indx_x,indx_z,indx_y)


            A[indx_z,indx_x] = A[indx_z,indx_x] + q_theta * v_z[indx_z] * np.dot(  moving_matrix[indx_x, indx_x: ] * g_vec , phi_Hs[indx_x:]  )

            A[indx_z,indx_z] = A[indx_z,indx_z] - q_theta * v_z[indx_z] * np.dot(  moving_matrix[indx_x, indx_x: ] * g_vec , phi_Hs[indx_x:]  )



    # Solve model
    return np.linalg.solve(A, b)


def prob_being_fired(indx_z, moving_matrix, para, theta, q_theta, phi_Fv, s_z):
    """
    The probability that an individual worker at firm indx_z is fired by the firm downsizing

    Parameters
    ----------
    indx_z: Index of the firm's productivity in z_grid

    moving_matrix: Matrix showing which transitions between states are possible (which manager jobs workers will accept)

    para: Dictionary of model

    theta: labor market tightness

    q_theta: Probability of a firm matching with a worker

    phi_Fw: PDF of new worker hires

    v_z: policy rule vacancy posting rate


    Returns
    -------
    Float: Probability of being fired
    """

    inner_intgr = np.empty(indx_z+1 )

    for indx_y in np.arange(0,(indx_z+1) ):

        inner_intgr[indx_y] =  np.dot( moving_matrix[ indx_y:(indx_z+1) , indx_z] , trans_prob_vec(indx_y,indx_z) )

    # Desired (abs) change in labor - num. workers who leave
    tmp = para['gamma'] * ( 1.0/para['sigma'] - 1.0 ) / ( para['alpha']*(1.0-1.0/para['sigma']) - 1.0 )    - s_z[indx_z] * theta * q_theta * np.dot(inner_intgr , phi_Fv[:(indx_z+1) ] )

    return max( (0,tmp) )


def get_new_vf_Pi(mu_z, moving_matrix, para, firm_rev, w, l_z, v_z, dz, q_theta, phi_Hs, z_grid=z_grid ):
    """
    Given the policy rules, managerial payment mu, (and holding the moving_matrix constant), compute the new value of
    being a firm at each grid point.

    We can write the firm's value function in matrix notation: A Pi = b

    Where Pi is an array of the value function of firms at each grid point, A is a (grid_size x grid_size) matrix,
    and b is an array.

    See documentation for exact equations being constructed here

    Parameters
    ----------
    mu_z: Vector of managerial payments to managers at each grid point

    moving_matrix: Matrix showing which transitions between states are possible (which manager jobs workers will accept)

    para: Dictionary of model

    firm_rev: revenue of a firm at each grid point

    w: Market clearing wage

    l_z: Number of workers hired by a firm at each grid point

    v_z: policy rule vacancy posting rate

    dz: step size between each productivity level in the grid space

    q_theta: Probability of a firm matching with a worker


    Returns
    -------
    Array: Value of being a worker at each grid point
    """

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



        for indx_x in np.arange(indx_z,grid_size):


            g_vec = np.empty(grid_size-indx_x)

            for indx_y in np.arange(indx_x,grid_size):
                g_vec[indx_y-indx_x] = trans_prob_indiv(indx_x,indx_z,indx_y)



            A[indx_z,indx_x] = A[indx_z,indx_x] + q_theta * v_z[indx_z] * np.dot( moving_matrix[indx_x, indx_x:] * g_vec  , phi_Hs[indx_x:] )

            A[indx_z,indx_z] = A[indx_z,indx_z] - q_theta * v_z[indx_z] * np.dot( moving_matrix[indx_x, indx_x:] * g_vec  , phi_Hs[indx_x:] )



    # Solve system of equations
    return np.linalg.solve(A, b)


def find_policy_rules(v_z, s_z, vf_M, vf_W, vf_Pi, para, q_theta, phi_Hs, phi_F, phi_Hw, l_z):
    """
    Given the value functions and distribution of productivity, solve for the policy rules s_z and v_z that maximise
    the value functions. This is carried out in each iteration of the main algorithm.

    Parameters
    ----------
    v_z: Initial guess of the vacancy posting rate

    s_z: Initial guess of the worker search intensity

    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    vf_Pi: value of a firm at each grid point

    para: Dictionary of model parameters

    phi_F: PDF of firm productivity

    phi_Hw: PDF of labor distribution

    l_z: Number of workers employed by a firm at each grid point


    Returns
    -------
    v_z = (array) vacancy posting rate

    s_z = (array) search intensity choice of a worker

    iter_count_pol = (float) number of iteration taken to converge
    """

    # Construct the matrix showing which managerial positions each worker will accept if matched with
    moving_matrix = create_move_matrix(vf_M, vf_W)


    for iter_count_pol in np.arange(0,max_pol_iter+1): # run loop until policy rules converge or max_pol_iter reached

        # Store the policy rules from the last loop so we can see if the policy rules have converged after this loop
        v_z_old = np.copy(v_z)
        s_z_old = np.copy(s_z)


        for indx_zn in np.arange(0,grid_size): # For each grid point

            # Update vacancy posting rate of firm
            #============================================================
            v_z_new = vac_post_rate(indx_zn, moving_matrix, para, q_theta, vf_Pi, phi_Hs) # new vacancy posting rate for firm indx_zn

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

                new_hires[indx_zj] = number_workers_hired(indx_zj,moving_matrix,phi_Fv, grid_size, l_z, q_theta, v_z,
                                                        phi_Hs, phi_F, s_z, theta, para)


            phi_Fw = new_hires / sum(new_hires) # Convert number of hires into PDF



            # Update Worker's search intensity
            #============================================================

            # Compute threshold
            search_threshold = s_thresh(indx_zn, moving_matrix , phi_Fv , para, theta, q_theta)

            search1 = search_intensity1(indx_zn, moving_matrix, phi_Fv, phi_Fw, para, vf_M, vf_W,  theta, q_theta)
            search2 = search_intensity2(indx_zn, moving_matrix, phi_Fv, para, vf_M, vf_W, theta, q_theta)

            s_z_new = (search1 < search_threshold) * search1 + (search2 > search_threshold) * search2


            s_z[indx_zn] = ff_pol * max([s_z_new, 0]) + (1.0 - ff_pol) * s_z_old[indx_zn]


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


        if iter_count_pol == max_pol_iter:

            print 'WARNING: Maximum number of policy iterations reached. Results may be inaccurate'

    return v_z, s_z


def compute_mu(vf_M, vf_W, q_theta, para, phi_Hs , z_grid=z_grid):
    """
    Find the managerial payment (mu) that makes the profit sharing equation hold true

    Parameters
    ----------
    vf_M: value of being a manager at each grid point

    vf_W: value of being a worker at each grid point

    q_theta: probability a firm matches with a worker

    para: Dictionary of model parameters

    phi_Hs: PDF of searching workers

    z_grid: state space grid


    Returns
    -------
    mu_z = (array) managerial payment
    """

    mu_z = np.empty(grid_size)

    for indx_z in np.arange(0,grid_size):
        if indx_z>0:
            dMdz = ( vf_M[indx_z] - vf_M[indx_z-1] ) / dz
        else:
            dMdz = 0

        inner_intgr = np.empty( grid_size - indx_z )

        for indx_y in np.arange(indx_z , grid_size ):

            inner_intgr[indx_y-indx_z] =  np.dot( moving_matrix_new[ indx_z:(indx_y+1) ,indx_y ] * ( vf_W[ indx_z:(indx_y+1) ] - vf_M[indx_z] )  , trans_prob_vec(indx_z, indx_y) )


        mu_z[indx_z] = ( para['r'] - para['gamma'] )*vf_M[indx_z] + para['gamma']*z_grid[indx_z] * dMdz - q_theta * v_z[indx_z] * np.dot( inner_intgr , phi_Hs[indx_z:] )

    return mu_z




########################################################################################################################
#   INITIAL GUESSES
########################################################################################################################
# Define the initial values for value functions and policy rules. This is needed to start the algorithm.



# Initial guesses for value functions
#----------------------------------------------------------
# The initial guess is based on the firms and workers continuing to receive their current income flow for all of eternity

# Compute the income flows etc
firm_rev    = firm_revenue( para, z_grid, phi_F )
w           = wages( para, z_grid, phi_F )
l_z         = labor_demand( para, z_grid, phi_F )


# Initial guess of the value function for firms (assuming managerial pay = 0)
vf_Pi   = ( firm_rev - w * l_z ) / ( para['r'] - para['gamma'] )

# Initial guess of the value function of workers
vf_W    = w / ( para['r'] - para['gamma'] ) * np.ones(grid_size)

# Initial guess of the value function of managers (based on manager profit sharing rule in the model)
vf_M = para['beta']/(1.0-para['beta'])*para['tau']*firm_rev + vf_W

# The initial guesses may not be that great. Therefore, we insure that the value of being a manager at a firm is at
# least as good as being a worker at the same firm
vf_M[vf_M < vf_W] = vf_W[vf_M < vf_W] + 0.000001




# Initial guesses for policy rules
#----------------------------------------------------------

s_z = np.ones(grid_size)  # Guess for worker search intensity

v_z = np.ones(grid_size)  # Guess for firm's vacancy posting rate

# Initial distribution of labor (phi_Hw) and worker's search intensity (phi_Hs) based on s_z
phi_Hw = l_z * phi_F / np.dot( l_z, phi_F )

phi_Hs = s_z  * phi_Hw / np.dot( s_z , phi_Hw  )



# Initial guess for labor market conditions
#----------------------------------------------------------

theta   = 1.0 / para['N']       # Guess for labor market tightness
q_theta = para['q_norm'] * theta ** para['q_cd']    # Probability of matching with a worker given theta




########################################################################################################################
#   EXECUTE THE ALGORITHM
########################################################################################################################

#===================================================================================================================
# Step zero: Given phi_F, update variable that only depend on it
#===================================================================================================================
l_z         = labor_demand(para, z_grid, phi_F)
phi_Hw      = dist_workers(para, z_grid, phi_F)
Y           = gdp(para, z_grid, phi_F)
w           = wages(para, z_grid, phi_F)
firm_rev    = firm_revenue(para, z_grid, phi_F)



for iter_count_vf in np.arange(1,max_vf_iter+1): # While the value functions are still different

    print iter_count_vf

    # Store old value functions so we can check if value functions have converged
    vf_M_old = np.copy(vf_M)
    vf_W_old = np.copy(vf_W)
    vf_Pi_old = np.copy(vf_Pi)



    #---------------------------------------------------------------------------------------------------------------
    # Step 1) Given phi_F and value functions, find policy rule
    #---------------------------------------------------------------------------------------------------------------
    v_z, s_z = find_policy_rules(v_z, s_z, vf_M, vf_W, vf_Pi, para, q_theta, phi_Hs, phi_F, phi_Hw, l_z)




    #---------------------------------------------------------------------------------------------------------------
    # Step 2) Given phi_F and policy rules, update value functions
    #---------------------------------------------------------------------------------------------------------------
    moving_matrix = create_move_matrix(vf_M,vf_W)

    # Update variables the are directly dependent upon the policy rules
    phi_Fv  = v_z * phi_F / np.dot( v_z , phi_F  )                      # Distribution of vacancy postings
    theta   = np.dot( v_z , phi_F ) / np.dot( para['N'] * s_z , phi_Hw) # Labor market tightness
    q_theta = para['q_norm'] * theta ** para['q_cd']                    # Prob of a firm matching with a worker
    phi_Hs  =   s_z * phi_Hw  / np.dot( s_z , phi_Hw )                  # Distribution of worker search intensity

    new_hires = np.empty(grid_size)
    for indx_zj in np.arange(0,grid_size): # For each firm on the grid, computer the number of new hires
            new_hires[indx_zj] = number_workers_hired(indx_zj,moving_matrix,phi_Fv, grid_size, l_z, q_theta, v_z,
                                                    phi_Hs, phi_F, s_z, theta, para)
    phi_Fw = new_hires / sum(new_hires) # Convert number of hires into PDF



    # Make new draw for vf_W
    vf_W_new = get_new_vf_W(moving_matrix, para, grid_size, dz, phi_Fv, firm_rev, theta, q_theta, s_z, v_z, phi_Fw, phi_Hs)

    # Compute implied values for vf_M
    vf_M_new = para['beta'] / (1.0-para['beta']) * para['tau'] * firm_rev + vf_W_new


    # Update moving matrix based on new value functions
    moving_matrix_new = create_move_matrix(vf_M_new,vf_W_new)


    # Given vf_M, solve for the managerial payment mu
    mu_z = compute_mu(vf_M_new, vf_W_new, q_theta, para, phi_Hs)


    # Solve for the firm's value function
    vf_Pi_new = get_new_vf_Pi(mu_z, moving_matrix_new, para, firm_rev, w, l_z, v_z, dz, q_theta, phi_Hs)


    # Check for convergence of the value function
    Pi_diff = max(abs(vf_Pi_new-vf_Pi_old))
    W_diff  = max(abs(vf_W_new-vf_W_old))
    M_diff  = max(abs(vf_M_new-vf_M_old))


    vf_diff = max([M_diff, W_diff, Pi_diff ])


    # Update value function estimates
    vf_Pi = ff_pol*vf_Pi_new + (1.0-ff_pol)*vf_Pi_old
    vf_M  = ff_pol*vf_M_new  + (1.0-ff_pol)*vf_M_old
    vf_W  = ff_pol*vf_W_new  + (1.0-ff_pol)*vf_W_old



    if vf_diff < tol_vf:
        break

    if iter_count_vf == max_vf_iter:
        print 'WARNING: Maximum number of value function iterations reached. Results may be inaccurate'


print "Model has completed running"