from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
import pdb

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''
    ##
    a=vehicle_params['a']
    b=vehicle_params['b']
    H=vehicle_params['H']
    L=vehicle_params['L']
    A=u[1]
    vc=u[0]/(1-np.tan(A)*H/L) 
    phi=ekf_state['x'][2]
    
    motion=np.zeros([3,1])
    motion[0]=dt*(vc*np.cos(phi)-vc/L*np.tan(A)*(a*np.sin(phi)+b*np.cos(phi)))
    motion[1]=dt*(vc*np.sin(phi)+vc/L*np.tan(A)*(a*np.cos(phi)-b*np.sin(phi)))
    motion[2]=dt*vc/L*np.tan(A)

    G=np.eye(3)                           
    G[0,2]=-dt*vc*np.sin(phi)-dt*vc/L*np.tan(A)*(a*np.cos(phi)-b*np.sin(phi))
    G[1,2]=dt*vc*np.cos(phi)-dt*vc/L*np.tan(A)*(a*np.sin(phi)+b*np.cos(phi))
    ##

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Performs the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    motion,G=motion_model(u, dt, ekf_state, vehicle_params)
    x=ekf_state['x'][0:3].copy()
  
    
    ekf_state['x'][0:3]=x+motion[:,0]
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    
    R_t = np.diag([sigmas['xy']**2,sigmas['xy']**2,sigmas['phi']**2])
    
    P = ekf_state['P'][0:3,0:3].copy()
    ekf_state['P'][0:3,0:3]=np.dot(np.dot(G,P),G.transpose())+R_t
    ekf_state['P']=slam_utils.make_symmetric(ekf_state['P'].copy())
    ###

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Performs a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    x=ekf_state["x"].copy()
    P=ekf_state["P"].copy()
    
    H = np.zeros([2, len(x)])
    H[0,0] = 1
    H[1,1] = 1
    
    R_t = (sigmas['gps']**2)*np.eye(2)
    S=P[0:2,0:2]+ R_t

    update = gps.reshape(2,1)-x[0:2].reshape(2,1)
    
    inv_S = slam_utils.invert_2x2_matrix(S)
    d = np.dot(np.dot(np.transpose(update),inv_S),update)
    
    if d<13.8:
        K=np.dot(P[:,0:2],inv_S)

        ekf_state['x'] = x+np.dot(K,update).flatten()
        ekf_state['x'][2]=slam_utils.clamp_angle(ekf_state['x'][2])

        ekf_state['P']=slam_utils.make_symmetric(np.dot((np.eye(len(x))-np.dot(K, H)),P))
        
    
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    x=ekf_state['x'].copy()
    
    x_v=x[3+2*landmark_id]
    y_v=x[4+2*landmark_id]
    
    zhat=np.zeros([2,1])
    zhat[0]=np.sqrt((x[0]-x_v)**2+(x[1]-y_v)**2)
    zhat[1]=slam_utils.clamp_angle(np.arctan2(y_v-x[1],x_v-x[0])-x[2])         
  
    ###
    
    H=np.zeros([2,len(x)])
    H[0,0] = (x[0]-x_v)/zhat[0]
    H[0,1] = (x[1]-y_v)/zhat[0]
    H[0,3+2*landmark_id] = -H[0,0]
    H[0,3+2*landmark_id+1] = -H[0,1]
    H[1,2] = -1
    H[1,0] = (y_v-x[1])/(zhat[0]**2)
    H[1,1] = (x[0]-x_v)/(zhat[0]**2)
    H[1,3+2*landmark_id] = -H[1,0]
    H[1,3+2*landmark_id+1] = -H[1,1]                         
    
    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initializes a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
   
    ###
    x = ekf_state['x'].copy()
    P = ekf_state['P'].copy()
    
    z_r = tree[0]
    z_beta = tree[1]
    
    x_l = x[0]+z_r*np.cos(z_beta+x[2])       
    y_l = x[1]+z_r*np.sin(z_beta+x[2])
    
    
    ekf_state['x']= np.append(x, np.array([x_l, y_l]))
    

    ####
    initial_size = P.shape[0]
    new_P = 200*np.eye(initial_size+2)
    new_P[:initial_size,:initial_size] = P
    
    ekf_state['P'] = slam_utils.make_symmetric(new_P)
    
    #####
    ekf_state["num_landmarks"]= ekf_state["num_landmarks"]+1
    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    P = ekf_state["P"].copy()
    n = ekf_state['num_landmarks']
    
    m = len(measurements)
    cost_matrix = np.zeros([m,n+1])
    cost_matrix[:,n] = chi2.ppf(0.99, 2)
    
    R = np.diag([sigmas['range']**2,sigmas['bearing']**2])
    
    
    
    for i in range(n):
        zhat, H = laser_measurement_model(ekf_state, i)

        S = np.dot(np.dot(H, P), H.transpose()) + R
        inv_S = slam_utils.invert_2x2_matrix(S)
        for j in range(m):
            update = np.asarray(measurements[j][0:2])-zhat.flatten()
            cost_matrix[j,i] = np.dot(np.dot(update.transpose(),inv_S),update)
            
    
    
    result = slam_utils.solve_cost_matrix_heuristic(cost_matrix)

    assoc = [0]*m
    for i,j in result:
        if j< ekf_state["num_landmarks"]:
            assoc[i] = j
        else:
            if min(cost_matrix[i,0:])<chi2.ppf(0.99, 2):
                assoc[i]=-2
            else:
                assoc[i]=-1
    
    
    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Performs a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    
    ###
    assoc = np.asarray(assoc)
    trees = np.asarray(trees)
    n= ekf_state['num_landmarks']
    if assoc.shape[0] != 0:
        new_tree_index = assoc == -1
        new_trees = trees[new_tree_index,:]
        for i in range(new_trees.shape[0]):
            initialize_landmark(ekf_state, new_trees[i,:])
            
        assoc[new_tree_index] = np.linspace(n, n+new_trees.shape[0]-1,new_trees.shape[0])
        Q = np.diag([sigmas['range']**2,sigmas['bearing']**2])
        exisiting_trees_index = assoc > -1
        
        
        exisiting_trees = trees[exisiting_trees_index,:]
        landmark_ids = assoc[exisiting_trees_index]
        
        for i in range(exisiting_trees.shape[0]):
            measure = exisiting_trees[i, 0:2].reshape(2,1)
            
            zhat,H = laser_measurement_model(ekf_state,int(landmark_ids[i]))        
            P = ekf_state['P'].copy()
            K = np.dot(np.dot(P,H.transpose()),slam_utils.invert_2x2_matrix(np.dot(np.dot(H,P),H.transpose())+Q))
            
            ekf_state['x'] = ekf_state['x']+np.dot(K,(measure-zhat))[:,0]
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            
            ekf_state['P'] = slam_utils.make_symmetric(np.dot((np.eye(P.shape[0])-np.dot(K,H)),P))
    
    ###

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)
       
        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": False

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
