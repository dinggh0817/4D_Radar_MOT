'''
GNN-PMB Tracker

Ref:
[1] J. Liu, L. Bai, Y. Xia, T. Huang, B. Zhu, and Q.-L. Han, “GNN-PMB: A Simple but Effective Online 3D Multi-Object Tracker without Bells and Whistles,”
IEEE Trans. Intell. Veh., pp. 1–14, 2022, doi: 10.1109/TIV.2022.3217490.
'''

import numpy as np
from numpy.linalg import inv, cholesky, det
import copy
from ismember import ismember

from scipy.linalg import block_diag, sqrtm
from matplotlib import pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

from ..utils.mhtdaClink import mhtda, allocateWorkvarsforDA
from ..utils.mhtdaClink import sparsifyByRow as sparsify

class GNN_PMB_Filter:

    def __init__(self, model): 
        self.model = model # use generated model which is configured for all parameters used in PMBM filter model for tracking the multi-targets.


    def preprocess_measurements(self, Z_k, is_the_first_frame, filter_predicted=None, ego_position=None):
        dim = self.model['dim']
        number_of_measurements_from_current_frame = Z_k['hwlxyza'].shape[1]
        used_measurements = Z_k['hwlxyza']
        if ego_position is None:
            ego_position = np.zeros(dim)

        if is_the_first_frame:
            number_of_surviving_previously_miss_detected_targets = 0
            number_of_surviving_previously_detected_targets = 0
        else:
            H = self.model['H_k'] # Measurement model
            meas_deviation =self.model['meas_deviation']
            detected_ellips_gating_threshold = self.model['detected_ellipsoidal_gating_threshold']
            undetected_ellips_gating_threshold = self.model['undetected_ellipsoidal_gating_threshold']

            number_of_surviving_previously_miss_detected_targets = len(filter_predicted['weightPois'])
            number_of_surviving_previously_detected_targets = len(filter_predicted['tracks'])

            '''
            Step 1.0. Meaasurement preprocessing
            1.0.1. Ellipsoidal gating, generate gating matrix for previously detected/undetected targets
            '''
            # Ellipsoidal gating for detected targets
            distance_meas_to_detected_targets = np.zeros([number_of_surviving_previously_detected_targets, number_of_measurements_from_current_frame])
            gating_matrix_of_detected_target = [[] for x in range(number_of_surviving_previously_detected_targets)]
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
                # Loop through all single target hypotheses belong to global hyptheses from previous frame. 
                for single_target_hypothesis_index_from_previous_frame in range(number_of_single_target_hypotheses_from_previous_frame):
                    if number_of_measurements_from_current_frame == 0:
                        gating_matrix_of_detected_target[previously_detected_target_index].append(np.zeros([0]).astype(int))
                    else:
                        gating_matrix_of_detected_target[previously_detected_target_index].append(np.zeros([number_of_measurements_from_current_frame]).astype(int))

                        mean_predict = (filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame])[0:dim,0]
                        # Calculate measurement noise covariance matrix at the center of each target
                        center_rotate_angle = np.arctan2(mean_predict[1] - ego_position[1], mean_predict[0] - ego_position[0])
                        center_rot_mat = np.array([
                            [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                            [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
                        ])
                        cov_center = np.diag([
                            meas_deviation[1]**2,
                            (np.sqrt(np.sum((mean_predict[0:2] - ego_position)**2)) * meas_deviation[0])**2
                        ])
                        rotated_cov_center = center_rot_mat @ cov_center @ center_rot_mat.T

                        covMat = H @ filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame] @ (H.T) + rotated_cov_center
                        invCovMat = inv((covMat + covMat.T)/2)
                        innovation = Z_k['hwlxyza'][3:3+dim, :] - H @ filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                        # Mahalanobis distance
                        dist = (innovation.T.dot(invCovMat)*innovation.T).sum(axis=1)
                        gating_matrix_of_detected_target[previously_detected_target_index][single_target_hypothesis_index_from_previous_frame][np.where(dist<detected_ellips_gating_threshold)[0]] = 1

                        euclid_dist = np.sqrt(np.sum(innovation**2, axis=0))
                        max_euclid_dist = self.model['max_gating_euclidean_distance']
                        gating_matrix_of_detected_target[previously_detected_target_index][single_target_hypothesis_index_from_previous_frame][np.where(euclid_dist>max_euclid_dist)[0]] = 0
                        distance_meas_to_detected_targets[previously_detected_target_index, :] = euclid_dist

            # Ellipsoidal gating for undetected targets (including new-born undetected tragets)
            distance_meas_to_miss_detected_targets = np.zeros([number_of_surviving_previously_miss_detected_targets, number_of_measurements_from_current_frame])
            gating_matrix_of_undetected_target = [[] for x in range(number_of_surviving_previously_miss_detected_targets)]
            for previously_undetected_target_index in range(number_of_surviving_previously_miss_detected_targets):
                if number_of_measurements_from_current_frame == 0:
                    gating_matrix_of_undetected_target[previously_undetected_target_index] = np.zeros([0]).astype(int)
                else:
                    gating_matrix_of_undetected_target[previously_undetected_target_index] = np.zeros([number_of_measurements_from_current_frame]).astype(int)

                    mean_predict = filter_predicted['meanPois'][previously_undetected_target_index][0:dim,0]
                    # Calculate measurement noise covariance matrix at the center of each target
                    center_rotate_angle = np.arctan2(mean_predict[1] - ego_position[1], mean_predict[0] - ego_position[0])
                    center_rot_mat = np.array([
                        [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                        [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
                    ])
                    cov_center = np.diag([
                        meas_deviation[1]**2,
                        (np.sqrt(np.sum((mean_predict[0:2] - ego_position)**2)) * meas_deviation[0])**2
                    ])
                    rotated_cov_center = center_rot_mat @ cov_center @ center_rot_mat.T

                    covMat = H @ filter_predicted['covPois'][previously_undetected_target_index] @ H.T + rotated_cov_center
                    invCovMat = inv((covMat + covMat.T)/2)
                    innovation = Z_k['hwlxyza'][3:3+dim, :] - H @ filter_predicted['meanPois'][previously_undetected_target_index]
                    # Mahalanobis distance
                    dist = (innovation.T.dot(invCovMat)*innovation.T).sum(axis=1)
                    gating_matrix_of_undetected_target[previously_undetected_target_index][np.where(dist<undetected_ellips_gating_threshold)[0]] = 1

                    euclid_dist = np.sqrt(np.sum(innovation**2, axis=0))
                    max_euclid_dist = self.model['max_gating_euclidean_distance']
                    gating_matrix_of_undetected_target[previously_undetected_target_index][np.where(euclid_dist>max_euclid_dist)[0]] = 0
                    distance_meas_to_miss_detected_targets[previously_undetected_target_index,:] = euclid_dist


        meas_preprocessed = {}
        meas_preprocessed['used_measurements'] = used_measurements
        meas_preprocessed['number_of_measurements_from_current_frame'] = number_of_measurements_from_current_frame
        meas_preprocessed['distance_meas_to_detected_targets'] = distance_meas_to_detected_targets
        meas_preprocessed['distance_meas_to_miss_detected_targets'] = distance_meas_to_miss_detected_targets
        meas_preprocessed['gating_matrix_of_detected_target'] = gating_matrix_of_detected_target
        meas_preprocessed['gating_matrix_of_undetected_target'] = gating_matrix_of_undetected_target

        return meas_preprocessed


    def predict(self, lag_time, filter_pruned, Z_k, ego_position=None):
        dim = self.model['dim']
        dim_of_state = self.model['dim_of_state']
        if ego_position is None:
            ego_position = np.zeros(dim)

        F = np.eye(dim_of_state, dtype=np.float64)
        I = lag_time*np.eye(dim, dtype=np.float64)
        F[0:dim, dim:dim+dim] = I  # Dynamic matrix
        # drive_noise_deviation = self.model["acceleration_deviation"]
        drive_noise_deviation1 = self.model["acceleration_deviation"]
        drive_noise_deviation2 = self.model["turn_rate_noise_deviation"]
        Q = drive_noise_deviation1**2 * np.array([
            [lag_time**3/3, 0, lag_time**2/2, 0],
            [0, lag_time**3/3, 0, lag_time**2/2,],
            [lag_time**2/2, 0, lag_time, 0],
            [0, lag_time**2/2, 0, lag_time]]) # Process noise covariance matrix
        Q_CT = block_diag(Q, drive_noise_deviation2**2*lag_time)

        birth_rate = self.model['birth_rate']

        number_of_surviving_previously_miss_detected_targets = len(filter_pruned['weightPois'])
        number_of_surviving_previously_detected_targets=len(filter_pruned['tracks'])

        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['hwlzaPois']=[] # Directly propagated states
        filter_predicted['idPois']=[]
        filter_predicted['max_idPois']=filter_pruned['max_idPois']
        # MBM Components data structure
        if number_of_surviving_previously_detected_targets > 0:
            filter_predicted['tracks'] = [{} for i in range(number_of_surviving_previously_detected_targets)]
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = copy.deepcopy(filter_pruned['globHyp'])
            filter_predicted['globHypWeight'] = copy.deepcopy(filter_pruned['globHypWeight'])
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                filter_predicted['tracks'][previously_detected_target_index]['eB']=[] # need to be filled in with prediction value                
                filter_predicted['tracks'][previously_detected_target_index]['meanB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['covB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['hwlzaB']=[] # need to be filled in with prediction value
                filter_predicted['tracks'][previously_detected_target_index]['idB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['log_weight_of_single_hypothesis']=\
                    copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['log_weight_of_single_hypothesis'])

        else:
            filter_predicted['tracks'] = []
            filter_predicted['max_idB']=filter_pruned['max_idB']
            filter_predicted['globHyp'] = []
            filter_predicted['globHypWeight'] = []
        """
        Step 1.1 : Prediction for surviving previously miss detected targets(i.e. the targets were undetected at previous frame and survive into current frame) by using PPP.
        """
        # Compute where it would have been should this track have been detected in previous step.
        if number_of_surviving_previously_miss_detected_targets > 0:
            for PPP_component_index in range(number_of_surviving_previously_miss_detected_targets):
                # Get data from previous frame
                weightPois_previous = filter_pruned['weightPois'][PPP_component_index]
                id_previous = filter_pruned['idPois'][PPP_component_index]
                meanPois_previous = filter_pruned['meanPois'][PPP_component_index]
                covPois_previous = filter_pruned['covPois'][PPP_component_index]
                hwlzaPois_previous = filter_pruned['hwlzaPois'][PPP_component_index]

                if dim_of_state == 4:
                    meanPois_predicted = F @ meanPois_previous
                    covPois_predicted = F @ covPois_previous @ F.T + Q
                    rot_mat = np.eye(2)
                elif dim_of_state == 5:
                    w = meanPois_previous[4,0]
                    t = lag_time
                    if w > 1e-3:
                        F_CT = np.array([
                            [1, 0, np.sin(w*t)/w,       -(1-np.cos(w*t))/w, 0],
                            [0, 1, (1-np.cos(w*t))/w,   np.sin(w*t)/w,      0],
                            [0, 0, np.cos(w*t),         -np.sin(w*t),       0],
                            [0, 0, np.sin(w*t),         np.cos(w*t),        0],
                            [0, 0, 0, 0, 1]
                        ])
                    else:
                        F_CT = F
                    meanPois_predicted = F_CT @ meanPois_previous
                    covPois_predicted = F_CT @ covPois_previous @ F_CT.T + Q_CT

                Ps = self.model['p_S']
                # distance = np.sqrt(np.sum((ego_position[0:dim] - (meanPois_predicted.squeeze())[0:dim])**2))
                # # TODO: Add configurable parameter
                # if distance > 40:
                #     Ps*=0.5
                # if distance > 60:
                #     Ps=0.1
                weightPois_predicted = Ps * weightPois_previous

                # Fill in the data structure
                filter_predicted['weightPois'].append(weightPois_predicted)
                filter_predicted['idPois'].append(id_previous)
                filter_predicted['meanPois'].append(meanPois_predicted) 
                filter_predicted['covPois'].append(covPois_predicted)
                filter_predicted['hwlzaPois'].append(hwlzaPois_previous)

        """
        Step 1.2 : Prediction for existing/surviving previously detected targets
        (i.e. targets were detected at previous frame and survive into current frame) by using Bernoulli components, or so called Multi-Bernoulli RFS.
        """
        if number_of_surviving_previously_detected_targets > 0:
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                for single_target_hypothesis_index_from_previous_frame in range(len(filter_pruned['tracks'][previously_detected_target_index]['eB'])):
                    # Get data from previous frame
                    eB_previous = filter_pruned['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                    idB_previous = filter_pruned['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]
                    meanB_previous = filter_pruned['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                    covB_previous = filter_pruned['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                    hwlzaB_previous = filter_pruned['tracks'][previously_detected_target_index]['hwlzaB'][single_target_hypothesis_index_from_previous_frame]
                    
                    if dim_of_state == 4:
                        meanB_predicted = F @ meanB_previous
                        covB_predicted = F @ covB_previous @ F.T + Q
                        rot_mat = np.eye(2)
                    elif dim_of_state == 5:
                        w = meanB_previous[4,0]
                        t = lag_time
                        if w > 1e-3:
                            F_CT = np.array([
                                [1, 0, np.sin(w*t)/w,       -(1-np.cos(w*t))/w, 0],
                                [0, 1, (1-np.cos(w*t))/w,   np.sin(w*t)/w,      0],
                                [0, 0, np.cos(w*t),         -np.sin(w*t),       0],
                                [0, 0, np.sin(w*t),         np.cos(w*t),        0],
                                [0, 0, 0, 0, 1]
                            ])
                        else:
                            F_CT = F
                        meanB_predicted = F_CT @ meanB_previous
                        covB_predicted = F_CT @ covB_previous @ F_CT.T + Q_CT

                    # Same prediction as PPP
                    Ps = self.model['p_S']
                    # distance = np.sqrt(np.sum((ego_position[0:dim] - (meanB_predicted.squeeze())[0:dim])**2))
                    # # TODO: Add configurable parameter
                    # if distance > 40:
                    #     Ps*=0.5
                    # if distance > 60:
                    #     Ps=0.1
                    eB_predicted = Ps * eB_previous

                    # Fill in the data structure                   
                    filter_predicted['tracks'][previously_detected_target_index]['eB'].append(eB_predicted)                    
                    filter_predicted['tracks'][previously_detected_target_index]['idB'].append(idB_previous)
                    filter_predicted['tracks'][previously_detected_target_index]['meanB'].append(meanB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['covB'].append(covB_predicted)
                    filter_predicted['tracks'][previously_detected_target_index]['hwlzaB'].append(hwlzaB_previous)
        
        """
        Step 1.4 : Prediction for new birth targets by using PPP.
        """
        meas_preprocessed = self.preprocess_measurements(Z_k, is_the_first_frame=False, filter_predicted=filter_predicted, ego_position=ego_position)

        meas_hwlxyza = meas_preprocessed['used_measurements']
        number_of_new_birth_targets = meas_preprocessed['number_of_measurements_from_current_frame']

        new_birth_minimum_distance = self.model["new_birth_minimum_distance"]
        # new_birth_minimum_distance = 0
        meas_indices = np.arange(number_of_new_birth_targets)
        if number_of_surviving_previously_detected_targets > 0:
            min_dist_meas_to_targets = np.min(meas_preprocessed['distance_meas_to_detected_targets'], axis=0)
            idx_birth_meas = np.greater(min_dist_meas_to_targets, new_birth_minimum_distance)
            meas_indices = [idx_meas for idx_meas in meas_indices if idx_birth_meas[idx_meas]]
            number_of_new_birth_targets = len(meas_indices)

        proposal_covariance_pos = self.model['position_total_covariance']
        proposal_deviation_vel = self.model['velocity_prior_deviation']
        proposal_deviation_turn_rate = 5*self.model['turn_rate_noise_deviation']
        meas_deviation = self.model["meas_deviation"]
        for new_birth_target_index in range(number_of_new_birth_targets):
            # Compute for the birth initiation
            weightPois_birth = birth_rate
            kinematic_birth = np.zeros([dim_of_state,1])
            proposal_mean = meas_hwlxyza[3:3+dim, [meas_indices[new_birth_target_index]]]
            kinematic_birth[0:dim,:] = proposal_mean[0:dim,:]
            hwlza_birth = meas_hwlxyza[[0,1,2,5,6], meas_indices[new_birth_target_index]]

            position_predict = proposal_mean[0:dim,0]
            # Calculate measurement noise covariance matrix at the center of each target
            center_rotate_angle = np.arctan2(position_predict[1], position_predict[0])
            center_rot_mat = np.array([
                [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
            ])
            cov_center = np.diag([
                meas_deviation[1]**2,
                (np.sqrt(np.sum((position_predict[0:2] - ego_position)**2)) * meas_deviation[0])**2
            ])
            rotated_cov_center = center_rot_mat @ cov_center @ center_rot_mat.T

            if dim_of_state == 4:
                cov_kinematic_birth = block_diag(proposal_covariance_pos + rotated_cov_center, proposal_deviation_vel**2*np.eye(dim))
            elif dim_of_state == 5:
                cov_kinematic_birth = block_diag(proposal_covariance_pos + rotated_cov_center, proposal_deviation_vel**2*np.eye(dim), proposal_deviation_turn_rate**2)

            # Fill in the data structure
            filter_predicted['weightPois'].append(weightPois_birth)  # Create the weight of PPP using the weight of the new birth PPP
            filter_predicted['idPois'].append(filter_predicted['max_idPois'] + 1 + new_birth_target_index)
            filter_predicted['meanPois'].append(kinematic_birth)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(cov_kinematic_birth)    # Create the variance of PPP using the variance of the new birth PPP
            filter_predicted['hwlzaPois'].append(hwlza_birth)
        filter_predicted['max_idPois'] += number_of_new_birth_targets
        filter_predicted['num_new_targets'] = number_of_new_birth_targets

        return filter_predicted, meas_preprocessed


    def predict_initial_step(self, Z_k, ego_position=None):
        """
        Compute the predicted intensity of new birth targets for the initial step (first frame).
        It has to be done separately because there is no input to initial step.
        There are other ways to implementate the initialization of the structure, this is just easier for the readers to understand.
        """
        dim = self.model['dim']
        dim_of_state = self.model['dim_of_state']
        birth_rate = self.model['birth_rate']
        if ego_position is None:
            ego_position = np.zeros(dim)

        # Create an empty dictionary filter_predicted which will be filled in by calculation and output from this function.
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['hwlzaPois']=[]

        filter_predicted['tracks'] = []
        filter_predicted['max_idB']=0
        filter_predicted['globHyp'] = []
        filter_predicted['globHypWeight'] = []
        filter_predicted['idPois']=[]

        meas_preprocessed = self.preprocess_measurements(Z_k, is_the_first_frame=False, filter_predicted=filter_predicted, ego_position=ego_position)

        meas_hwlxyza = meas_preprocessed['used_measurements']
        number_of_new_birth_targets = meas_preprocessed['number_of_measurements_from_current_frame']

        proposal_covariance_pos = self.model['position_total_covariance']
        proposal_deviation_vel = self.model['velocity_prior_deviation']
        proposal_deviation_turn_rate = 5*self.model['turn_rate_noise_deviation']
        meas_deviation = self.model["meas_deviation"]
        for new_birth_target_index in range(number_of_new_birth_targets):
            # Compute for the birth initiation
            weightPois_birth = birth_rate
            kinematic_birth = np.zeros([dim_of_state,1])
            proposal_mean = meas_hwlxyza[3:3+dim, [new_birth_target_index]]
            kinematic_birth[0:dim,:] = proposal_mean[0:dim,:]
            hwlza_birth = meas_hwlxyza[[0,1,2,5,6], new_birth_target_index]

            position_predict = proposal_mean[0:dim,0]
            # Calculate measurement noise covariance matrix at the center of each target
            center_rotate_angle = np.arctan2(position_predict[1], position_predict[0])
            center_rot_mat = np.array([
                [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
            ])
            cov_center = np.diag([
                meas_deviation[1]**2,
                (np.sqrt(np.sum((position_predict[0:2] - ego_position)**2)) * meas_deviation[0])**2
            ])
            rotated_cov_center = center_rot_mat @ cov_center @ center_rot_mat.T

            if dim_of_state == 4:
                cov_kinematic_birth = block_diag(proposal_covariance_pos + rotated_cov_center, proposal_deviation_vel**2*np.eye(dim))
            elif dim_of_state == 5:
                cov_kinematic_birth = block_diag(proposal_covariance_pos + rotated_cov_center, proposal_deviation_vel**2*np.eye(dim), proposal_deviation_turn_rate**2)

            # Fill in the data structure
            filter_predicted['weightPois'].append(weightPois_birth)  # Create the weight of PPP using the weight of the new birth PPP
            filter_predicted['idPois'].append(new_birth_target_index)
            filter_predicted['meanPois'].append(kinematic_birth)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(cov_kinematic_birth)    # Create the variance of PPP using the variance of the new birth PPP
            filter_predicted['hwlzaPois'].append(hwlza_birth)
        filter_predicted['max_idPois'] = number_of_new_birth_targets
        filter_predicted['num_new_targets'] = number_of_new_birth_targets

        return filter_predicted, meas_preprocessed


    """
    Step 2: Update Section V-C of [1]
    2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are still undetected at current frame, just update the weight of PPP but mean 
            and covarince remains same.
    2.2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are now associated with detections(detected) at current frame, corresponding 
            Bernoulli RFS is converted from PPP normally by updating (PPP --> Bernoulli) for each of them.
    2.2.2. For the measurements(detections) which can not be in the gating area of any previously miss detected target or any new birth target(both represented by PPP), corresponding 
            Bernoulli RFS is created by filling most of the parameters of this Bernoulli as zeors (create Bernoulli with zero existence probability, stands for detection is originated 
            from clutter) for each of them.
    2.3.1. For the previously detected targets which are now undetected at current frame, just update the eB of the distribution but mean and covarince remains same for each of them.
    2.3.2. For the previously detected targets which are now associated with detection(detected) at current frame, the parameters of the distribution is updated for each of them.  
    """
    def update(self, filter_predicted, meas_preprocessed, ego_position=None, filter_previous_step=None):
        if filter_previous_step is None:
            return filter_predicted
        # Get pre-defined parameters.
        dim = self.model['dim']
        dim_of_state = self.model['dim_of_state']
        if ego_position is None:
            ego_position = np.zeros(dim)
        Pd =self.model['p_D'] # probability for detection
        clutter_intensity = self.model['clutter_intensity']

        H = self.model['H_k'] # measurement model
        meas_deviation = self.model['meas_deviation']

        # Get components information from filter_predicted.
        number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets = len(filter_predicted['weightPois'])
        number_of_detected_targets_from_previous_frame = len(filter_predicted['tracks'])
        number_of_global_hypotheses_from_previous_frame = len(filter_predicted['globHyp'])

        # Get preprocessed measurements
        used_measurements = meas_preprocessed['used_measurements']
        number_of_used_measurements = used_measurements.shape[1]
        gating_matrix_of_detected_target = meas_preprocessed['gating_matrix_of_detected_target']
        gating_matrix_of_undetected_target = meas_preprocessed['gating_matrix_of_undetected_target']

        # Initialize data structures for filter_update
        filter_updated = {}
        filter_updated['weightPois'] = []
        filter_updated['meanPois'] = []
        filter_updated['covPois'] = []
        filter_updated['hwlzaPois'] = []
        filter_updated['idPois'] = []
        filter_updated['max_idPois'] = filter_predicted['max_idPois']

        # Updated date structure initialization
        if number_of_detected_targets_from_previous_frame==0:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            if number_of_used_measurements == 0:
                filter_updated['tracks'] = []
                filter_updated['max_idB'] = 0
            else: 
                filter_updated['tracks']=[{} for n in range(number_of_used_measurements)] # Initiate the data structure with right size of dictionaries
                for i in range(number_of_used_measurements): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][i]['eB']= []
                    filter_updated['tracks'][i]['meanB']= []
                    filter_updated['tracks'][i]['covB']= []
                    filter_updated['tracks'][i]['hwlzaB']= []
                    filter_updated['tracks'][i]['idB']=[]
                    filter_updated['tracks'][i]['log_weight_of_single_hypothesis']= []
        else:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            filter_updated['tracks']=[{} for n in range(number_of_detected_targets_from_previous_frame + number_of_used_measurements)]
            # Initiate data structure for indexing 0 to (number of detected target index)
            for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                filter_updated['tracks'][previously_detected_target_index]['eB'] = []
                filter_updated['tracks'][previously_detected_target_index]['meanB'] = []
                filter_updated['tracks'][previously_detected_target_index]['covB'] = []
                filter_updated['tracks'][previously_detected_target_index]['hwlzaB'] = []
                filter_updated['tracks'][previously_detected_target_index]['idB'] = []
                filter_updated['tracks'][previously_detected_target_index]['log_weight_of_single_hypothesis'] = []

            # Initializing data structure for index from (number of previously detected targets) to (number of previosly detected targets + number of measurements)
            for i in range(number_of_used_measurements): # Initialte the dictionary with empty list.
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['eB'] = []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['meanB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['covB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['hwlzaB']= []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['idB'] = []
                filter_updated['tracks'][number_of_detected_targets_from_previous_frame + i]['log_weight_of_single_hypothesis'] = []

        # Store the temporary update results in these data structures
        track_new = [{} for x in range(number_of_used_measurements)]
        for i in range(number_of_used_measurements):
            # Create single target hypothesis for non-existent target
            track_new[i]['meas_idx'] = [-1]
            track_new[i]['eB'] = [0]
            track_new[i]['meanB'] = [np.zeros([dim_of_state, 1])]
            track_new[i]['covB'] = [np.ones([dim_of_state, dim_of_state])]
            track_new[i]['hwlzaB'] = [np.zeros([5, 1])]
            track_new[i]['log_weight_of_single_hypothesis'] = [0]
            track_new[i]['idB'] = [-1]

        track_upd = [{} for x in range(number_of_detected_targets_from_previous_frame)]
        for i in range(number_of_detected_targets_from_previous_frame):
            local_hypo_num = len(filter_predicted['tracks'][i]['eB'])
            track_upd[i]['meas_idx'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['eB'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['meanB'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['covB'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['hwlzaB'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['log_weight_of_single_hypothesis'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['idx_of_single_target_hypothesis'] = [[] for x in range(local_hypo_num)]
            track_upd[i]['idB'] = [[] for x in range(local_hypo_num)]

        track_new_measurementwise = [{} for x in range(number_of_used_measurements)]
        for i in range(number_of_used_measurements):
            track_new_measurementwise[i]['meas_idx'] = []
            track_new_measurementwise[i]['eB'] = []
            track_new_measurementwise[i]['meanB'] = []
            track_new_measurementwise[i]['covB'] = []
            track_new_measurementwise[i]['hwlzaB'] = []
            track_new_measurementwise[i]['log_weight_of_single_hypothesis'] = []
            track_new_measurementwise[i]['idB'] = []

        """
        Step 2.1. for update:  Missed Detection Hypothesis for PPP Components.
        Update step for "the targets which were miss detected previosly and still remain undetected at current frame, and new birth targets got undetected
        at current frame. Remain PPP. 
        """
        # Miss detected target and new birth target are modelled by using Poisson Point Process(PPP). This is the same as the miss detected target modelling part in [2].
        # Notice the reason mean and covariance remain the same is because if there is no detection, there would be no update.
        Qd = 1-Pd # The probability of sensor failing to detect the target
        for PPP_component_index in range(number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets):
            # Get predicted data
            weightPois_predicted = filter_predicted['weightPois'][PPP_component_index]
            meanPois_predicted = filter_predicted['meanPois'][PPP_component_index]
            covPois_predicted = filter_predicted['covPois'][PPP_component_index]
            hwlzaPois_predicted = filter_predicted['hwlzaPois'][PPP_component_index]
            idPois_predicted = filter_predicted['idPois'][PPP_component_index]

            if PPP_component_index < (number_of_miss_detected_targets_from_previous_frame_and_new_birth_targets - filter_predicted['num_new_targets']):
                weightPois_updated = Qd * weightPois_predicted
            else:
                weightPois_updated = weightPois_predicted
            
            # Fill in the data structure, no actual GGIW update for miss detected targets
            filter_updated['weightPois'].append(weightPois_updated)
            filter_updated['meanPois'].append(meanPois_predicted)
            filter_updated['covPois'].append(covPois_predicted)
            filter_updated['hwlzaPois'].append(hwlzaPois_predicted)
            filter_updated['idPois'].append(idPois_predicted)
        filter_updated['max_idPois'] = filter_predicted['max_idPois']

        """
        Step 2.2. for update: Generate number_of_used_measurements new Bernoulli components(some of new Bernoulli components are converted from PPP, others are 
                    created originally.). Section V-C1 of [1]
        2.2.1: Convert Poisson Point Processes to Bernoulli RFSs. Update the targets which were miss detected previosly but now get detected at current frame, by updating with 
                    the valid measurements within gating area.
        2.2.2: Create new Bernoulli RFSs. For the measurement not falling into gating area of any PPP component, it is assumed to be originated from clutter. Create a Bernoulli 
                    RSF by filling parameters with zeros for each of them anyway for data structure purpose.
        """

        for meas_idx in range(number_of_used_measurements):
            if len(gating_matrix_of_undetected_target) > 0:
                # Check if the measurement is in the gate of any PPP components
                ppp_idx = np.where(np.array(gating_matrix_of_undetected_target)[:, meas_idx])[0]
            else:
                ppp_idx = np.array([])

            '''
            2.2.1: If current measurement is associated with PPP component(previously miss-detected target or new birth target), use this measurement to update the target, 
                    thus convert corresponding PPP into Bernoulli RFS.
            '''
            if ppp_idx.size > 0:
                weight_PPP_component_predicted = np.array([filter_predicted['weightPois'][idx] for idx in ppp_idx.astype(int)])
                mean_PPP_component_predicted = np.array([filter_predicted['meanPois'][idx] for idx in ppp_idx.astype(int)])
                cov_PPP_component_predicted = np.array([filter_predicted['covPois'][idx] for idx in ppp_idx.astype(int)])
                this_measurement = used_measurements[3:3+dim, meas_idx].reshape(dim, -1)
                this_hwlza = used_measurements[[0,1,2,5,6], meas_idx]

                # Kalman update
                rotated_cov_center = np.zeros([ppp_idx.size, 2, 2])
                for idx in range(ppp_idx.size):
                    mean_predict = mean_PPP_component_predicted[idx,0:dim,0] - ego_position
                    # Calculate measurement noise covariance matrix at the center of each target
                    center_rotate_angle = np.arctan2(mean_predict[1], mean_predict[0])
                    center_rot_mat = np.array([
                        [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                        [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
                    ])
                    cov_center = np.diag([
                        meas_deviation[1]**2,
                        (np.sqrt(np.sum(mean_predict**2)) * meas_deviation[0])**2
                    ])
                    rotated_cov_center[idx,:,:] = center_rot_mat @ cov_center @ center_rot_mat.T
                epsilon = this_measurement - H @ mean_PPP_component_predicted
                # Compute for innovation covariance S: H * P-_current * H' + R
                S = H @ cov_PPP_component_predicted @ H.T + rotated_cov_center
                S = (S + S.transpose([0,2,1])) / 2
                # Kalman gain
                K = cov_PPP_component_predicted @ H.T @ inv(S)
                mean_PPP_component_updated = mean_PPP_component_predicted + K @ epsilon
                cov_PPP_component_updated = cov_PPP_component_predicted - K @ H @ cov_PPP_component_predicted

                log_likelihood_updated = -dim/2 * np.log(2*np.pi) - 0.5*np.log(det(S)) - 0.5*(epsilon.transpose([0,2,1]) @ inv(S) @ epsilon).squeeze() + np.log(Pd)

                # Normalize updated PPP weights
                log_weights = np.log(weight_PPP_component_predicted).squeeze() + log_likelihood_updated.squeeze()
                if log_weights.size <= 1:
                    log_sum_weights_temp = log_weights
                    log_weights_normalized = np.zeros([log_weights.size])
                else:
                    log_weights_sort_descend_idx = np.flip(np.argsort(log_weights))
                    log_sum_weights_temp = log_weights[log_weights_sort_descend_idx[0]] + \
                        np.log(1 + np.sum(np.exp(log_weights[log_weights_sort_descend_idx[1:]] - log_weights[log_weights_sort_descend_idx[0]])))
                    log_weights_normalized = log_weights - log_sum_weights_temp

                # Add all clutter case, calculate existence probability
                log_weights = np.append(log_sum_weights_temp, np.log(clutter_intensity))
                log_weights_sort_descend_idx = np.flip(np.argsort(log_weights))
                log_sum_weights = log_weights[log_weights_sort_descend_idx[0]] + \
                    np.log(1 + np.sum(np.exp(log_weights[log_weights_sort_descend_idx[1:]] - log_weights[log_weights_sort_descend_idx[0]])))
                log_weights_normalized_temp = log_weights - log_sum_weights

                # existence_probability_updated = 1
                existence_probability_updated = np.exp(log_sum_weights_temp - log_sum_weights)
                
                nonzero_idx = np.where(np.exp(log_weights_normalized)>0)[0]
                nonzero_weights = np.exp(log_weights_normalized[nonzero_idx])
                sum_nonzero_weights = np.sum(nonzero_weights)
                num_nonzero = len(nonzero_idx)
                if num_nonzero == 0:
                    print("Likelihood calculation error! No PPP associates with measurement.")

                # Kinematic merge
                mean_kinematic_merged = np.sum(mean_PPP_component_updated[nonzero_idx] * nonzero_weights.reshape(-1,1,1), axis=0)
                cov_kinematic_merged = np.zeros(cov_PPP_component_updated[0].shape)
                for idx in range(nonzero_idx.size):
                    x_diff = (mean_PPP_component_updated[nonzero_idx[idx]] - mean_kinematic_merged).reshape(-1,1)
                    cov_kinematic_merged += nonzero_weights[idx] * (cov_PPP_component_updated[idx] + x_diff.dot(x_diff.T))

                # Fill in the data structure
                track_new_measurementwise[meas_idx]['eB'].append(existence_probability_updated) 
                track_new_measurementwise[meas_idx]['meanB'].append(mean_kinematic_merged)
                track_new_measurementwise[meas_idx]['covB'].append(cov_kinematic_merged)
                track_new_measurementwise[meas_idx]['hwlzaB'].append(this_hwlza)
                track_new_measurementwise[meas_idx]['log_weight_of_single_hypothesis'].append(log_sum_weights)
                track_new_measurementwise[meas_idx]['meas_idx'].append(meas_idx)
                track_new_measurementwise[meas_idx]['idB'].append(filter_predicted['max_idB'] + meas_idx + 1)

            else:
                '''
                2.2.2
                If there is not any PPP component(previously miss-detected target or new birth target) could be associated with the current measurement, assume this measurement is originated from clutter. 
                We still need to create a Bernoulli component for it, since we need to guarantee that every measurement generates a Bernoulli RFS.
                The created Bernoulli component has existence probability zero (denote it is clutter). It will be removed by pruning.
                '''

                mean_updated = np.zeros([dim_of_state,1])
                cov_updated = np.eye(dim_of_state)
                hwlza_updated = np.zeros([5,1])
                log_sum_weights = np.log(clutter_intensity)

                # Fill in the data structure
                track_new_measurementwise[meas_idx]['eB'].append(0) 
                track_new_measurementwise[meas_idx]['meanB'].append(mean_updated)
                track_new_measurementwise[meas_idx]['covB'].append(cov_updated)
                track_new_measurementwise[meas_idx]['hwlzaB'].append(hwlza_updated)
                track_new_measurementwise[meas_idx]['log_weight_of_single_hypothesis'].append(log_sum_weights)
                track_new_measurementwise[meas_idx]['meas_idx'].append(meas_idx)
                track_new_measurementwise[meas_idx]['idB'].append(filter_predicted['max_idB'] + meas_idx + 1)

        """
        Step 2.3. for update: Section V-C2 of [1]
        Update for targets which got detected at previous frame.
        """
       
        for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
            
            number_of_single_target_hypotheses_from_previous_frame = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
        
            # Loop through all single target hypotheses (STH) belong to global hyptheses from previous frame. 
            for single_target_hypothesis_index_from_previous_frame in range(number_of_single_target_hypotheses_from_previous_frame):
                # Get the data from filter_predicted
                mean_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame]
                cov_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame]
                hwlza_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['hwlzaB'][single_target_hypothesis_index_from_previous_frame]
                eB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame]
                idB_single_target_hypothesis_predicted = filter_predicted['tracks'][previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame]

                """
                Step 2.3.1. for update: Undetected Hypothesis
                Update the targets got detected previously but get undetected at current frame.
                """
                # Compute for missed detection hypotheses
                probability_for_track_exist_but_undetected = 1 - Pd
                temp = 1 - eB_single_target_hypothesis_predicted + eB_single_target_hypothesis_predicted * probability_for_track_exist_but_undetected
                log_weight_undetected = np.log(temp)
                eB_undetected = eB_single_target_hypothesis_predicted * probability_for_track_exist_but_undetected / temp
                
                # Fill in the data structure
                track_upd[previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame].append(eB_undetected)
                track_upd[previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame].append(mean_single_target_hypothesis_predicted)
                track_upd[previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame].append(cov_single_target_hypothesis_predicted)
                track_upd[previously_detected_target_index]['hwlzaB'][single_target_hypothesis_index_from_previous_frame].append(hwlza_single_target_hypothesis_predicted)
                track_upd[previously_detected_target_index]['log_weight_of_single_hypothesis'][single_target_hypothesis_index_from_previous_frame].append(log_weight_undetected)
                track_upd[previously_detected_target_index]['meas_idx'][single_target_hypothesis_index_from_previous_frame].append(-1)
                track_upd[previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame].append(idB_single_target_hypothesis_predicted)

                """
                Step 2.3.2. for update:
                Update the targets got detected previously and still get detected at current frame.
                Beware what we do here is to update all the possible single target hypotheses and corresponding cost value for every single target hypothesis(each 
                target-measurement possible association pair). The single target hypotheses which can happen at the same time will form a global hypothesis(joint 
                event), and all the global hypotheses will be formed exhaustively later by using part of "all the possible single target hypotheses". 
                """
                for meas_idx in range(number_of_used_measurements):
                    # Check if the measurement is in the gate of the corresponding single target hypothesis (STH)
                    if gating_matrix_of_detected_target[previously_detected_target_index][single_target_hypothesis_index_from_previous_frame][meas_idx] == 1:
                        this_measurement = used_measurements[3:3+dim, meas_idx].reshape(dim, 1)
                        this_measurement_hwlza = used_measurements[[0,1,2,5,6], meas_idx].squeeze()

                        # Kalman update
                        mean_predict = mean_single_target_hypothesis_predicted[0:dim,0] - ego_position
                        # Calculate measurement noise covariance matrix at the center of each target
                        center_rotate_angle = np.arctan2(mean_predict[1], mean_predict[0])
                        center_rot_mat = np.array([
                            [np.cos(center_rotate_angle),-np.sin(center_rotate_angle)],
                            [np.sin(center_rotate_angle), np.cos(center_rotate_angle)]
                        ])
                        cov_center = np.diag([
                            meas_deviation[1]**2,
                            (np.sqrt(np.sum(mean_predict**2)) * meas_deviation[0])**2
                        ])
                        rotated_cov_center = center_rot_mat @ cov_center @ center_rot_mat.T
                        epsilon = this_measurement - H @ mean_single_target_hypothesis_predicted
                        S = H @ cov_single_target_hypothesis_predicted @ H.T + rotated_cov_center
                        S = (S + S.T) / 2
                        K = cov_single_target_hypothesis_predicted @ H.T @ inv(S)
                        mean_updated = mean_single_target_hypothesis_predicted + K @ epsilon
                        cov_updated = cov_single_target_hypothesis_predicted - K @ H @ cov_single_target_hypothesis_predicted

                        log_likelihood_updated = -dim/2 * np.log(2*np.pi) - 0.5*np.log(det(S)) - 0.5*(epsilon.T @ inv(S) @ epsilon).squeeze() + np.log(Pd)

                        eB_updated = 1
                        log_weight_updated = float(log_likelihood_updated + np.log(eB_single_target_hypothesis_predicted))

                        # Fill in the data structure
                        track_upd[previously_detected_target_index]['eB'][single_target_hypothesis_index_from_previous_frame].append(eB_updated)
                        track_upd[previously_detected_target_index]['meanB'][single_target_hypothesis_index_from_previous_frame].append(mean_updated)
                        track_upd[previously_detected_target_index]['covB'][single_target_hypothesis_index_from_previous_frame].append(cov_updated)
                        track_upd[previously_detected_target_index]['hwlzaB'][single_target_hypothesis_index_from_previous_frame].append(this_measurement_hwlza)
                        track_upd[previously_detected_target_index]['log_weight_of_single_hypothesis'][single_target_hypothesis_index_from_previous_frame].append(log_weight_updated)
                        track_upd[previously_detected_target_index]['meas_idx'][single_target_hypothesis_index_from_previous_frame].append(meas_idx)
                        track_upd[previously_detected_target_index]['idB'][single_target_hypothesis_index_from_previous_frame].append(idB_single_target_hypothesis_predicted)

        # Create updated single target hypotheses for the first detection of undetected targets
        for meas_idx in range(number_of_used_measurements):
            # Create new single target hypothesis
            track_new[meas_idx]['meas_idx'].append(meas_idx)
            track_new[meas_idx]['eB'] += track_new_measurementwise[meas_idx]['eB']
            track_new[meas_idx]['meanB'] += track_new_measurementwise[meas_idx]['meanB']
            track_new[meas_idx]['covB'] += track_new_measurementwise[meas_idx]['covB']
            track_new[meas_idx]['hwlzaB'] += track_new_measurementwise[meas_idx]['hwlzaB']
            track_new[meas_idx]['log_weight_of_single_hypothesis'] += track_new_measurementwise[meas_idx]['log_weight_of_single_hypothesis']
            track_new[meas_idx]['idB'] += track_new_measurementwise[meas_idx]['idB']
        number_of_new_tracks = number_of_used_measurements

        # Update local hypothesis trees
        for target_idx in range(number_of_detected_targets_from_previous_frame):
            idx = 0
            for single_target_hypo_idx in range(len(track_upd[target_idx]['eB'])):
                filter_updated['tracks'][target_idx]['eB'] += track_upd[target_idx]['eB'][single_target_hypo_idx]
                filter_updated['tracks'][target_idx]['meanB'] += track_upd[target_idx]['meanB'][single_target_hypo_idx]
                filter_updated['tracks'][target_idx]['covB'] += track_upd[target_idx]['covB'][single_target_hypo_idx]
                filter_updated['tracks'][target_idx]['hwlzaB'] += track_upd[target_idx]['hwlzaB'][single_target_hypo_idx]
                filter_updated['tracks'][target_idx]['log_weight_of_single_hypothesis'] += track_upd[target_idx]['log_weight_of_single_hypothesis'][single_target_hypo_idx]
                filter_updated['tracks'][target_idx]['idB'] += track_upd[target_idx]['idB'][single_target_hypo_idx]
                # Use an extra variable to record the index of each new single target hypothesis in local hypothesis tree
                meas_num = len(track_upd[target_idx]['meas_idx'][single_target_hypo_idx])
                track_upd[target_idx]['idx_of_single_target_hypothesis'][single_target_hypo_idx] = (np.arange(0,meas_num) + idx).tolist()
                idx += meas_num
        for new_track_idx in range(number_of_new_tracks):
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['eB'] += track_new[new_track_idx]['eB']
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['meanB'] += track_new[new_track_idx]['meanB']
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['covB'] += track_new[new_track_idx]['covB']
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['hwlzaB'] += track_new[new_track_idx]['hwlzaB']
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['log_weight_of_single_hypothesis'] += track_new[new_track_idx]['log_weight_of_single_hypothesis']
            filter_updated['tracks'][number_of_detected_targets_from_previous_frame + new_track_idx]['idB'] += track_new[new_track_idx]['idB']

        if len(filter_updated['tracks']) > 0:
            filter_updated['max_idB'] = np.max([np.max(x['idB']) for x in filter_updated['tracks']])
        else:
            filter_updated['max_idB'] = filter_predicted['max_idB']

        """
        Step 2.4. for update:
        Update Global Hypotheses as described in section V-C3 in [1].
        The objective of global hypotheses is to select k optimal single target hypotheses to propogate towards the next step.
        """
        log_weight_of_global_hypothesis_in_log_format=[]
        for global_hypo_index_from_previous_frame in range(max(1, number_of_global_hypotheses_from_previous_frame)):
            log_weight_for_missed_detection_hypotheses = 0
            if number_of_global_hypotheses_from_previous_frame == 0:
                global_hypo_indices = []
            else:
                global_hypo_indices = filter_predicted['globHyp'][global_hypo_index_from_previous_frame]
                # missed detection hypothesis is the first data generated under any previous single target hypothesis
                for target_idx in range(number_of_detected_targets_from_previous_frame):
                    if global_hypo_indices[target_idx] > -1:
                        log_weight_for_missed_detection_hypotheses += (track_upd[target_idx]['log_weight_of_single_hypothesis'][global_hypo_indices[target_idx]])[0]

            # If there is no measurement, all the targets are misdetected
            if number_of_used_measurements == 0:
                table_update = [int(x) for x in -np.ones(number_of_detected_targets_from_previous_frame + number_of_new_tracks)]
                if len(global_hypo_indices) > 0:
                    for target_idx in range(number_of_detected_targets_from_previous_frame):
                        if global_hypo_indices[target_idx] > -1:
                            table_update[target_idx] = (track_upd[target_idx]['idx_of_single_target_hypothesis'][global_hypo_indices[target_idx]])[0]
                for idx in range(number_of_new_tracks):
                    table_update[number_of_detected_targets_from_previous_frame + idx] = 0
                if number_of_global_hypotheses_from_previous_frame == 0:
                    filter_updated['globHypWeight'].append(np.exp(log_weight_for_missed_detection_hypotheses))
                else:
                    filter_updated['globHypWeight'].append(np.exp(log_weight_for_missed_detection_hypotheses)
                                                            * filter_predicted['globHypWeight'][global_hypo_index_from_previous_frame])
                filter_updated['globHyp'].append(table_update)
                continue

            '''
            Step 2.4.1 Generate Cost Matrix For Each Global Hypothesis Index from previous frame.
            '''
            # Construct cost matrix
            cost_matrix_log = -np.inf*np.ones((number_of_detected_targets_from_previous_frame + number_of_new_tracks, number_of_used_measurements))
            optimal_associations_all = []
            '''
            Step 2.4.1.1 Fill in cost_matrix with regard to the detected tracks under each of measurement partitions.
            '''
            if number_of_global_hypotheses_from_previous_frame > 0:
                for previously_detected_target_index in range(number_of_detected_targets_from_previous_frame):
                    '''
                    1. read out the previous single target hypothesis index specified by the global hypothesis
                    '''
                    single_target_hypothesis_index_specified_by_previous_step_global_hypothesis =\
                        filter_predicted['globHyp'][global_hypo_index_from_previous_frame][previously_detected_target_index] 
                    if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis != -1: # if this track exist                                
                        # Fill in the cost matrix
                        # Only get data that is generated under the corresponding single_target_hypothesis_index_from_previous_frame
                        '''
                        2. get the indices of current single target hypotheses generated under this global hypothesis 
                        '''
                        locA, locB = ismember(
                            track_upd[previously_detected_target_index]['meas_idx'][single_target_hypothesis_index_specified_by_previous_step_global_hypothesis], np.arange(number_of_used_measurements))
                        if len(locB) > 0:
                            cost_matrix_log[previously_detected_target_index, locB] = \
                                np.array(track_upd[previously_detected_target_index]['log_weight_of_single_hypothesis'][single_target_hypothesis_index_specified_by_previous_step_global_hypothesis])[locA] \
                                - track_upd[previously_detected_target_index]['log_weight_of_single_hypothesis'][single_target_hypothesis_index_specified_by_previous_step_global_hypothesis][0]
                                        
            '''
            Step 2.4.1.2 Fill in the cost matrix with regard to the newly initiated tracks
            '''
            for track_index in range(number_of_new_tracks):
                locA, locB = ismember(track_new[track_index]['meas_idx'], np.arange(number_of_used_measurements))
                if len(locB) > 0:
                    cost_matrix_log[number_of_detected_targets_from_previous_frame + track_index, locB] = \
                        np.array(track_new[track_index]['log_weight_of_single_hypothesis'])[locA]

            '''
            Step 2.4.2 Genereta the K (which varies each frame) optimal option based on cost matrix
            1. Remove -Inf rows and columns for performing optimal assignment. We take them into account for indexing later.
                Columns that have only one value different from Inf are not fed into Murty either.
            2. Use Murty to get the k-best optimal options
            3. Add the cost of the misdetection hypothesis to the cost matrix
            4. Add back the removed infinity options
            '''
            indices_of_cost_matrix_with_valid_elements = 1 - np.isinf(cost_matrix_log)
            # We only use Murty on non-exclusive part of the cost matrix
            indices_of_meas_non_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])>1]
            if len(indices_of_meas_non_exclusive)>0:
                indices_of_tracks_non_exclusive=[x for x in range(len(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_meas_non_exclusive])))
                                                if sum(np.transpose([indices_of_cost_matrix_with_valid_elements[:,x] for x in indices_of_meas_non_exclusive])[x]>0)]
                cost_matrix_log_non_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_meas_non_exclusive]))[indices_of_tracks_non_exclusive]
            else:
                indices_of_tracks_non_exclusive = []
                cost_matrix_log_non_exclusive = []

            # The exclusive part of the cost matrix
            indices_of_meas_exclusive = [x for x in range(len(indices_of_cost_matrix_with_valid_elements[0])) if sum(indices_of_cost_matrix_with_valid_elements[:,x])==1]
            if len(indices_of_meas_exclusive) > 0:
                indices_of_tracks_exclusive = [np.argmax(indices_of_cost_matrix_with_valid_elements[:,x]) for x in indices_of_meas_exclusive]
                cost_matrix_log_exclusive = np.array(np.transpose([cost_matrix_log[:,x] for x in indices_of_meas_exclusive]))[indices_of_tracks_exclusive]
            else:
                indices_of_tracks_exclusive = []
                cost_matrix_log_exclusive = []
                    
            if len(cost_matrix_log_non_exclusive)==0:
                association_vector=np.zeros(number_of_used_measurements).astype(int)
                for index_of_idx, idx in enumerate(indices_of_meas_exclusive):
                    association_vector[idx]=indices_of_tracks_exclusive[index_of_idx]
                optimal_associations_all.append(association_vector)
                cost_for_optimal_associations_non_exclusive = [0] 
            else:
                if number_of_global_hypotheses_from_previous_frame == 0:
                    k_best_global_hypotheses_under_this_previous_global_hypothesis = self.model['maximum_number_of_global_hypotheses']
                else:
                    k_best_global_hypotheses_under_this_previous_global_hypothesis = \
                        int(np.ceil(self.model['maximum_number_of_global_hypotheses']*filter_predicted['globHypWeight'][global_hypo_index_from_previous_frame]))
                # subtract positive maximum cost to reduce miss assignment
                cost_matrix = -np.transpose(cost_matrix_log_non_exclusive)
                max_cost = np.max(cost_matrix.reshape(-1)[np.logical_not(np.isinf(cost_matrix.reshape(-1)))]) + 100
                # if max_cost < 0:
                #     max_cost = 0
                cost_matrix = cost_matrix - max_cost
                nrows, ncolumns = cost_matrix.shape
                nsolutions = k_best_global_hypotheses_under_this_previous_global_hypothesis # find the lowest-cost associations

                # sparse cost matrices only include a certain number of elements the rest are implicitly infinity
                cost_matrix_sparse = sparsify(cost_matrix, np.min([ncolumns, 30]))
                # mhtda is set up to potentially take multiple input hypotheses for both rows and columns input hypotheses specify a subset of rows or columns.
                # In this case, we just want to use the whole matrix.
                row_priors = np.ones((1, nrows), dtype=np.bool8)
                col_priors = np.ones((1, ncolumns), dtype=np.bool8)
                # Each hypothesis has a relative weight too. These values don't matter if there is only one hypothesis...
                row_prior_weights = np.zeros(1)
                col_prior_weights = np.zeros(1)
                # The mhtda function modifies preallocated outputs rather than allocating new ones. This is slightly more efficient for repeated use
                # within a tracker. The cost of each returned association:
                out_costs = np.zeros(nsolutions)
                # The row-column pairs in each association:
                # Generally there will be less than nrows+ncolumns pairs in an association. The unused pairs are currently set to (-2, -2)
                out_associations = np.zeros((nsolutions, nrows+ncolumns, 2), dtype=np.int32)
                # variables needed within the algorithm (a C function sets this up):
                workvars = allocateWorkvarsforDA(nrows, ncolumns, nsolutions)
                # run murty
                err = mhtda(cost_matrix_sparse,
                    row_priors, row_prior_weights, col_priors, col_prior_weights,
                    out_associations, out_costs, workvars)

                optimal_associations_non_exclusive = []
                cost_for_optimal_associations_non_exclusive = []
    
                # notice that k_best might be larger than the maximum number of options generated by murty
                if err != 0:
                    nsolutions = np.where(out_costs < 100000)[0][-1] + 1
                for solution in range(nsolutions):
                    # display row-column matches, not row misses or column misses
                    association = out_associations[solution]
                    association_matches = association[(association[:,0]>=0) & (association[:,1]>=0)]
                    if len(association_matches) == nrows:
                        association_array = -np.ones(nrows).astype(int)
                        association_array[association_matches[:,0]] = association_matches[:,1]
                        optimal_associations_non_exclusive.append(association_array.tolist())
                        cost_for_optimal_associations_non_exclusive.append(out_costs[solution] + max_cost*len(association_matches))

                # Optimal indices without removing Inf rows
                # the exclusive associations are placed at its appropriate position.
                # the indices of measurements non exclusive/exclusive are the lookup take for this procedure
                optimal_associations_all = -1*np.ones((len(optimal_associations_non_exclusive), number_of_used_measurements)).astype(int)
                for ith_optimal_option_index, ith_optimal_association_vector in enumerate(optimal_associations_non_exclusive):
                    # First handle the case where there are duplicated associations
                    for idx_of_non_exclusive_matrix, ith_optimal_track_idx in enumerate(ith_optimal_association_vector):
                        # find out the actual index through the lookup vectors
                        actual_meas_idx = indices_of_meas_non_exclusive[idx_of_non_exclusive_matrix]
                        actual_track_idx = indices_of_tracks_non_exclusive[ith_optimal_track_idx]
                        optimal_associations_all[ith_optimal_option_index][actual_meas_idx] = actual_track_idx
                    # Then handle the case wehre there are single association
                    for idx_of_exclusive_matrix, actual_meas_idx in enumerate(indices_of_meas_exclusive):
                        actual_track_idx = indices_of_tracks_exclusive[idx_of_exclusive_matrix]
                        optimal_associations_all[ith_optimal_option_index][actual_meas_idx] = actual_track_idx
                    
            # Compute for cost fixed from cost matrix exclusive assocition.
            # this need to be added back to the weight of global hypothesis because ealier, we deleted this part out.
            log_weight_of_exclusive_assosications = 0
            for row_index in range(len(cost_matrix_log_exclusive)):
                log_weight_of_exclusive_assosications += cost_matrix_log_exclusive[row_index][row_index]

            for ith_optimal_option in range(len(optimal_associations_all)):
                if number_of_global_hypotheses_from_previous_frame > 0:
                    log_weight_of_global_hypothesis_in_log_format.append(
                        -cost_for_optimal_associations_non_exclusive[ith_optimal_option] + log_weight_for_missed_detection_hypotheses +
                        log_weight_of_exclusive_assosications + np.log(filter_predicted['globHypWeight'][global_hypo_index_from_previous_frame])) # The global weight associated with this hypothesis
                else:
                    log_weight_of_global_hypothesis_in_log_format.append(
                        -cost_for_optimal_associations_non_exclusive[ith_optimal_option] + log_weight_for_missed_detection_hypotheses +
                        log_weight_of_exclusive_assosications) # The global weight associated with this hypothesis
    
            '''
            Step 2.4.3 Generate the new global hypothesis based on the cost matrix of this previous global hypothesis
            '''
            globHyp = -np.ones([len(optimal_associations_all), number_of_detected_targets_from_previous_frame + number_of_new_tracks]).astype(int)
            for association_idx in range(len(optimal_associations_all)):
                # Go through each association in a given assignment
                for meas_idx in range(number_of_used_measurements):
                    # Check if detected targets or undetected targets
                    current_association = (optimal_associations_all[association_idx][meas_idx]).tolist()
                    if current_association > -1:
                        if current_association < number_of_detected_targets_from_previous_frame:
                            # Find the index of the corresponding measurement
                            globHyp[association_idx, current_association] = \
                                np.array(track_upd[current_association]['idx_of_single_target_hypothesis'][global_hypo_indices[current_association]])[
                                    track_upd[current_association]['meas_idx'][global_hypo_indices[current_association]] == np.array(meas_idx)
                                ]
                        else:
                            # Find the index of the corresponding measurement
                            idx = np.where(track_new[current_association - number_of_detected_targets_from_previous_frame]['meas_idx'] == np.array(meas_idx))[0].tolist()
                            if len(idx) > 0:
                                globHyp[association_idx, current_association] = idx[0]
                    
                # Go through each unassociated track
                unassigned_track_indices = np.ones(number_of_detected_targets_from_previous_frame + number_of_new_tracks)
                unassigned_track_indices[optimal_associations_all[association_idx]] = 0
                unassigned_track_indices = np.where(unassigned_track_indices != 0)[0]
                for unassigned_idx in range(len(unassigned_track_indices)):
                    if unassigned_track_indices[unassigned_idx] < number_of_detected_targets_from_previous_frame:
                        if len(global_hypo_indices) > 0:
                            temp_global_hypo_idx = global_hypo_indices[unassigned_track_indices[unassigned_idx]]
                        else:
                            temp_global_hypo_idx = -1
                        if temp_global_hypo_idx > -1:
                            globHyp[association_idx, unassigned_track_indices[unassigned_idx]] = \
                                (track_upd[unassigned_track_indices[unassigned_idx]]['idx_of_single_target_hypothesis'][temp_global_hypo_idx])[0]
                        else:
                            globHyp[association_idx, unassigned_track_indices[unassigned_idx]] = -1
                    else:
                        globHyp[association_idx, unassigned_track_indices[unassigned_idx]] = 0

            for idx, ith_optimal_global_hypothesis in enumerate(globHyp.astype(int).tolist()):
                filter_updated['globHyp'].append(ith_optimal_global_hypothesis)

        # Normalize weights
        log_weights = np.array(log_weight_of_global_hypothesis_in_log_format)
        if log_weights.size <= 1:
            log_sum_weights_temp = log_weights
            log_weights_normalized = np.zeros([log_weights.size])
        else:
            log_weights_sort_descend_idx = np.flip(np.argsort(log_weights))
            log_sum_weights_temp = log_weights[log_weights_sort_descend_idx[0]] + \
                np.log(1 + np.sum(np.exp(log_weights[log_weights_sort_descend_idx[1:]] - log_weights[log_weights_sort_descend_idx[0]])))
            log_weights_normalized = log_weights - log_sum_weights_temp

        for ith_log_weight in log_weights_normalized:
            filter_updated['globHypWeight'].append(np.exp(ith_log_weight))

        return filter_updated

    """
    Step 3: State Estimation. Section VI of [1]
    Firstly, obtain the only global hypothesis with the "maximum weight" from remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm). 
    Then the state extraction is based on this only global hypothesis. Sepecifically, there are three ways to obtain this only global hypothesis:
    Option 1. The only global hypothesis is obtained via maximum globHypWeight: maxmum_global_hypothesis_index = argmax(globHypWeight).
    Option 2. First, compute for cardinality. Then compute weight_new according to cardinality. Finally, obtain the maximum only global hypothesis via this new weight via argmax(weight_new).
    Option 3. Generate deterministic cardinality via a fixed eB threshold. Then compute weight_new and argmax(weight_new) the same way as does Option 2.  
    """
    def extractStates(self, filter_updated, thld=-1):
        # Get data
        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypotheses_at_current_frame = len(globHypWeight)
        
        # Initiate datastructure
        state_estimate = {}
        mean = []
        hwlza = [] 
        eB_list = []
        id = []

        if thld == -1:
            thld = self.model['eB_estimation_threshold']

        if number_of_global_hypotheses_at_current_frame>0: # If there are valid global hypotheses
            highest_weight_global_hypothesis_index = np.argmax(globHypWeight) # get he index of global hypothesis with largest weight
            highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index] # get the global hypothesis with largest weight
            for track_index in range(len(highest_weight_global_hypothesis)): # number of tracks.
                single_target_hypothesis_specified_by_global_hypothesis=int(highest_weight_global_hypothesis[track_index]) # Get the single target hypothesis index.
                if single_target_hypothesis_specified_by_global_hypothesis > -1: # If the single target hypothesis exists
                    eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_specified_by_global_hypothesis]
                    if eB > thld: # if the existence probability is greater than the threshold
                        mean_updated = filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis_specified_by_global_hypothesis].squeeze()
                        hwlza_updated = filter_updated['tracks'][track_index]['hwlzaB'][single_target_hypothesis_specified_by_global_hypothesis].squeeze()
                        id_updated = filter_updated['tracks'][track_index]['idB'][single_target_hypothesis_specified_by_global_hypothesis]

                        eB_list.append(eB)
                        mean.append(mean_updated)
                        hwlza.append(hwlza_updated)
                        id.append(id_updated)

        state_estimate['mean'] = mean
        state_estimate['hwlza'] = hwlza
        state_estimate['id'] = id
        state_estimate['eB'] = eB_list

        return state_estimate
    
    """
    Step 4: Pruning
    4.1. Prune the Poisson part by discarding components whose weight is below a threshold.
    4.2. Prune the global hypothese by discarding components whose weight is below a threshold.
    4.3. Prune Multi-Bernoulli RFS:
    4.3.1. Mark single target hypothese whose existence probability is below a threshold.
    4.3.2. Based on the marks of previous step, delete tracks whose single target hypothesis are all below the threshold. 
    4.3.2. Remove Bernoulli components do not appear in the remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm).
            
    By doing this, only the single target hypotheses belong to the k best global hypotheses will be left, propogated to next frame as "root" to generate more
    single target hypotheses at next frame.
    In other words, more than one global hypotheses(i.e. the k best global hypothese) will be propagated into next frame as "base" to generate 
    more global hypotheses for next frame. This is why people claim that PMBM is a MHT like filter(in MHT, the multiple hypotheses are propogated 
    from previous frame to current frame thus generating more hypotheses based the "base" multiple hypotheses from previous frame, and the best 
    hypothesis is selected (like GNN) among all the generated hypotheses at current frame.
    """ 
    
    def prune(self, filter_updated):
        # Get pre-defined parameters.
        dim = self.model['dim']
        # initiate filter_pruned as a copy of filter_updated
        filter_pruned = copy.deepcopy(filter_updated)

        # extract pertinent data from the dictionary
        weightPois=copy.deepcopy(filter_updated['weightPois'])
        global_hypothesis_weights=copy.deepcopy(filter_updated['globHypWeight'])
        globHyp=copy.deepcopy(filter_updated['globHyp'])
        maximum_number_of_global_hypotheses = self.model['maximum_number_of_global_hypotheses']
        eB_threshold = self.model['eB_threshold']
        Poisson_threshold = self.model['T_pruning_Pois']
        MBM_threshold = self.model['T_pruning_MBM']
        """
        Step 4.1.1
        Prune the Poisson part by discarding components whose weight is below a threshold.
        """
        # if weight is smaller than the threshold, remove the Poisson component
        indices_to_remove_poisson=[index for index, value in enumerate(weightPois) if value<Poisson_threshold]
        for offset, idx in enumerate(indices_to_remove_poisson):
            del filter_pruned['weightPois'][idx-offset]
            del filter_pruned['idPois'][idx-offset]
            del filter_pruned['meanPois'][idx-offset]
            del filter_pruned['covPois'][idx-offset]
            del filter_pruned['hwlzaPois'][idx-offset]
        """
        Step 4.1.2
        Reduce the Poisson mixture (Removed)
        """
        if len(filter_pruned['weightPois']) > 1:
            filter_pruned['max_idPois'] = np.max(filter_pruned['idPois'])
        else:
            filter_pruned['max_idPois'] = -1

        """
        Step 4.2.
        Pruning Global Hypothesis
        Any global hypothese with weights smaller than the threshold would be pruned away. 
        """
        # only keep global hypothesis whose weight is larger than the threshold
        indices_to_keep_global_hypotheses=[index for index, value in enumerate(global_hypothesis_weights) if value>MBM_threshold]
        weights_after_pruning_before_capping=[global_hypothesis_weights[x] for x in indices_to_keep_global_hypotheses]
        globHyp_after_pruning_before_capping=[globHyp[x] for x in indices_to_keep_global_hypotheses]
        # negate the list first because we require the descending order instead of acending order.
        weight_after_pruning_negative_value = [-x for x in weights_after_pruning_before_capping]
        # If after previous step there are still more global hypothesis than desirable:
        # Pruning components so that there is at most a maximum number of components.
        # get the indices for global hypotheses in descending order.
        index_of_ranked_global_hypothesis_weights_in_descending_order=np.argsort(weight_after_pruning_negative_value) # the index of elements in ascending order
        if len(weights_after_pruning_before_capping)>maximum_number_of_global_hypotheses:
            # cap the list with maximum_number_of_global_hypotheses
            indices_to_keep_global_hypotheses_capped = index_of_ranked_global_hypothesis_weights_in_descending_order[:maximum_number_of_global_hypotheses]
        else:
            indices_to_keep_global_hypotheses_capped=index_of_ranked_global_hypothesis_weights_in_descending_order[:len((weights_after_pruning_before_capping))]
        
        
        globHyp_after_pruning = [copy.deepcopy(globHyp_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped] 
        weights_after_pruning = [copy.deepcopy(weights_after_pruning_before_capping[x]) for x in indices_to_keep_global_hypotheses_capped]
        # normalize weight of global hypotheses
        weights_after_pruning=[x/np.sum(weights_after_pruning) for x in weights_after_pruning]
        globHyp_after_pruning=np.array(globHyp_after_pruning)
        weights_after_pruning=np.array(weights_after_pruning)

        if len(globHyp_after_pruning)>0:

            """
            Step 4.3.1.
            Mark Bernoulli components whose existence probability is below a threshold
            Notice it is just to mark those components with existance probability that is below the threshold.
            We don't just simply remove those tracks, since even for elements with small existance probability,
            it is still possible that it is part of the k most optimal global hypotheses.
            Notice that this is the only path that would lead to a path deletion.
            """
            
            for track_index in range(len(filter_pruned['tracks'])):     
                # Get the indices for single target hypotheses that is lower than the threshold.
                indices_of_single_target_hypotheses_to_be_marked=[index for index,value in enumerate(filter_pruned['tracks'][track_index]['eB']) if value < eB_threshold]
                # If there is a single target hypothesis that should be removed but is part of global hypotheses
                #global_hypothesis_for_this_track_single_target_hypothesis_to_be_marked = []
                for single_target_hypothesis_to_be_marked_idx in indices_of_single_target_hypotheses_to_be_marked:
                    # check each element of the single target hypothese made up for the global hypotheses
                    for index_of_single_target_hypothesis_in_global_hypothesis,single_target_hypothesis_in_global_hypothesis in enumerate(globHyp_after_pruning[:,track_index]):
                        # if this single target hypothesis that is below the threshold is present at this global hypothesis
                        if single_target_hypothesis_in_global_hypothesis==single_target_hypothesis_to_be_marked_idx:
                            # mark it with -1, which would be utilized in the next step to initate track deletion.
                            globHyp_after_pruning[:,track_index][index_of_single_target_hypothesis_in_global_hypothesis]=-1
            # if the column vector sums to 0 means this track does not participate in any global hypothesis
            """
            Step 4.3.2.
            Remove tracks(Bernoulli components) that do not take part in any global hypothesis 
            """  
            # check if all the single target hypothesis under this track that participates in the global hypothesis
            # has existence probability that is below the threhold
            if len(globHyp_after_pruning) > 0:
                tracks_to_be_removed = [x for x in range(len(globHyp_after_pruning[0])) if np.sum(globHyp_after_pruning[:,x]) == -len(globHyp_after_pruning)]
            else:
                tracks_to_be_removed=[]
            if len(tracks_to_be_removed)>0:
                # delete the tracks whose single target hypothese has existence probability below threshold
                for offset, track_index_to_be_removed in enumerate(tracks_to_be_removed):
                    del filter_pruned['tracks'][track_index_to_be_removed-offset]
                # after the track deletion, delete the corresponding column vectors from global hypothesis matrix.
                globHyp_after_pruning = np.delete(globHyp_after_pruning, tracks_to_be_removed, axis=1)
            for track_index in range(len(filter_pruned['tracks'])): # notice that the number of tracks has changed
                """
                Step 4.3.3.
                Remove single-target hypotheses in each track (Bernoulli component) that do not belong to any global hypothesis.
                """  
                single_target_hypothesis_indices_to_be_removed = []            
                number_of_single_target_hypothesis =len(filter_pruned['tracks'][track_index]['eB'])
                
                # read out the single target hypothese participate in the global hypotheses
                valid_single_target_hypothesis_for_this_track = globHyp_after_pruning[:,track_index]
                # if a single target hypothesis does not participate in any global hypothesis
                # then it would be removed
               
                for single_target_hypothesis_index in range(number_of_single_target_hypothesis):
                    # if this single target hypothesis does not particulate in the global hypotheses
                    if single_target_hypothesis_index not in valid_single_target_hypothesis_for_this_track:
                        # add it to the to be removed list.
                        single_target_hypothesis_indices_to_be_removed.append(single_target_hypothesis_index)
                # if there are single target hypothese to be removed
                if len(single_target_hypothesis_indices_to_be_removed)>0:
                    # remove the single target hypotheses from the data structure
                    for offset, index in enumerate(single_target_hypothesis_indices_to_be_removed):
                        del filter_pruned['tracks'][track_index]['eB'][index-offset]
                        del filter_pruned['tracks'][track_index]['meanB'][index-offset]
                        del filter_pruned['tracks'][track_index]['covB'][index-offset]
                        del filter_pruned['tracks'][track_index]['hwlzaB'][index-offset]
                        del filter_pruned['tracks'][track_index]['idB'][index-offset]
                        del filter_pruned['tracks'][track_index]['log_weight_of_single_hypothesis'][index-offset]
                """
                Step 4.3.4.
                Adjust the global hypothesis indexing if there are deletions of single target hypotheses. 
                For instance, in the global hypothesis, the current single target hypothesis is indexed as 5.
                But during previous steps, single target hypothesis 3, 6 is pruned away,
                therefore, the remaining single target hypothesis number 5 need to be adjusted to 4. 
                """ 
                # after the deletion, readjust the indexing system
                if len(single_target_hypothesis_indices_to_be_removed)>0:
                    for global_hypothesis_index, global_hypothesis_vector in enumerate(globHyp_after_pruning):
                        # find out how many single target hypothesis are deleted before this single target hypothesis
                        single_target_hypothesis_specified_by_the_global_hypothesis = global_hypothesis_vector[track_index]
                        # the index of removed single target hypothesis before the end point
                        single_target_hypotheses_removed_before_this_single_taget_hypothesis = \
                            [x for x in single_target_hypothesis_indices_to_be_removed if x < single_target_hypothesis_specified_by_the_global_hypothesis]
                        # adjust the indexing system by the length of removed terms before this element.
                        subtraction = len(single_target_hypotheses_removed_before_this_single_taget_hypothesis)
                        globHyp_after_pruning[global_hypothesis_index][track_index] -= subtraction
      
            """
            Step 4.3.5.
            Merge the duplicated global hypothese if there are any. 
            """ 
            # After the previous steps of pruning, 
            # there can be duplication amongst the remaining global hypotheses. 
            # so the solution is to merged those duplications by adding their weights 
            # and leave only one element. 
    
            # get the unique elements
            globHyp_unique, indices= np.unique(globHyp_after_pruning, axis=0, return_index = True)
            # get the indices of duplication
            duplicated_indices = [x for x in range(len(globHyp_after_pruning)) if x not in indices]
            # check if there are any duplication.
            if len(globHyp_unique)!=len(globHyp_after_pruning): #There are duplicate entries
                weights_unique=np.zeros(len(globHyp_unique))
                for i in range(len(globHyp_unique)):
                    # fill in the weight of unique elements
                    weights_unique[i] = weights_after_pruning[indices[i]]
                    for j in duplicated_indices:
                        if list(globHyp_after_pruning[j]) == list(globHyp_unique[i]):
                            # add the weight of duplications to its respective unique elements.
                            weights_unique[i]+= weights_after_pruning[j]
    
                globHyp_after_pruning=globHyp_unique
                weights_after_pruning=weights_unique
                weights_after_pruning=weights_after_pruning/sum(weights_after_pruning)
        else:
            """
            If the global hypothesis table is empty, remove empty tracks
            """ 
            for track_index in range(len(filter_pruned['tracks'])):     
                if (len(filter_pruned['tracks'][track_index]['eB']) == 1) & (filter_pruned['tracks'][track_index]['eB'] == 0):
                    del filter_pruned['tracks'][track_index]
        
        filter_pruned['globHyp']=globHyp_after_pruning.tolist()
        filter_pruned['globHypWeight']=weights_after_pruning.tolist()
        return filter_pruned
