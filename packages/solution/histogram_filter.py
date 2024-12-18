# start by importing some things we will need
import numpy as np
from math import floor, sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal
from dt_state_estimation.lane_filter.types import Segment, SegmentColor, SegmentPoint


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief


# Now let's define the predict function


def histogram_predict(belief, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief

    # You may find the following code useful to find the current best heading estimate:
    maxids = np.unravel_index(belief_in.argmax(), belief_in.shape)
    phi_max = grid_spec['phi_min'] + (maxids[1] + 0.5) * grid_spec['delta_phi']

    
    # AHZ: Grid parameters
    #    1- 'd_min' and 'd_max' specify the range of possible positions (d)
    #    2- 'phi_min' and 'phi_max' specify the range of possible orientations (phi).
    #    3- 'd_resolution' and 'phi_resolution' tell us how fine the grid is.

    alpha = 2 * np.pi / robot_spec['encoder_resolution']
    
    linear_displacement_wheel_left = robot_spec['wheel_radius'] * left_encoder_ticks * alpha
    linear_displacement_wheel_right = robot_spec['wheel_radius'] * right_encoder_ticks * alpha
    linear_displacement_robot_origin = (linear_displacement_wheel_left + linear_displacement_wheel_right) / 2
    angular_displacement_robot_origin = (linear_displacement_wheel_right - linear_displacement_wheel_left) / robot_spec['wheel_baseline']

    # Compute the current position (d_t) and orientation (phi_t)
    v = linear_displacement_robot_origin
    w = angular_displacement_robot_origin

    # propagate each centroid
    d_t = grid_spec["d"] + v
    phi_t = grid_spec["phi"] + w

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            # If belief[i,j] there was no mass to move in the first place
            if belief[i, j] > 0:
                # Now check that the centroid of the cell wasn't propagated out of the allowable range
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue

                # AHZ: Find the new cell index (i_new, j_new) corresponding to the updated position and orientation.
                # This is because the belief grid is discrete. So, we need to convert this continuous position and orientation (d_t, phi_t)
                # into discrete grid indices (i_new, j_new) that correspond to a cell in the belief grid. The way, I do this is:
                # 1- 'd_new - d_min': Shifts the new position (d_t) so that it starts at the grid's minimum value.
                # 2- Dividing by 'd_resolution': Converts the continuous position into the number of grid cells from the start.
                # 3- Rounding: Ensures the value falls exactly on a grid cell index.
                i_new = int((d_t[i, j] - grid_spec['d_min']) / grid_spec["delta_d"])
                j_new = int((phi_t[i, j] - grid_spec['phi_min']) / grid_spec["delta_phi"])

                # AHZ: Transfer the probability mass from the old cell (i, j) to the new cell (i_new, j_new)
                p_belief[i_new, j_new] += belief[i, j]
    
    
    # AHZ: Real robot movements are never perfect; there’s always some noise in the motion.
    # To account for this uncertainty, the function smooths the updated belief grid (p_belief) by applying a Gaussian blur. 
    # This spreads the probability across nearby cells.
    # 'cov_mask' is used to model uncertainty in the robot’s motion by applying a Gaussian blur, 
    # which spreads probability across neighboring cells.
    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    # AHZ: Normalizing the Belief -> After the propagation and smoothing steps, the belief grid may no longer sum to 1.
    # The final step is to normalize the belief grid, ensuring that all probabilities add up to 1.
    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)

    # AHZ: The function returns the predicted belief grid after considering the robot’s movement and adding process noise.
    # This updated belief shows where the robot is most likely to be, given its previous belief and motion.
    return belief


# We will start by doing a little bit of processing on the segments to remove anything that is
# behing the robot (why would it be behind?) or a color not equal to yellow or white


def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != SegmentColor.WHITE and segment.color != SegmentColor.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments

def generate_vote(segment, road_spec):
    p1 = segment.points[0].as_array()
    p2 = segment.points[1].as_array()
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

    n_hat = np.array([-t_hat[1], t_hat[0]])
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == SegmentColor.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane

            d_i = -d_i

            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == SegmentColor.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i



def generate_measurement_likelihood(segments, road_spec, grid_spec):
    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        # So now we have d_i and phi_i which correspond to the estimate of the distance
        # from the center and the angle relative to the center generated by the single
        # segment under consideration

        # AHZ: Once we have the voted pose/orientation corresponding to a segment, we need to convert this continuous position and orientation (d_t, phi_t)
        # into discrete grid indices (i_new, j_new) that correspond to a cell in the belief grid. The way, I do this is:
        # 1- 'd_new - d_min': Shifts the new position (d_t) so that it starts at the grid's minimum value.
        # 2- Dividing by 'd_resolution': Converts the continuous position into the number of grid cells from the start.
        # 3- Rounding: Ensures the value falls exactly on a grid cell index.
        i = int((d_i - grid_spec['d_min']) / grid_spec["delta_d"])
        j = int((phi_i - grid_spec['phi_min']) / grid_spec["delta_phi"])

        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        # Combine the prior belief and the measurement likelihood to get the posterior belief
        # We need to normalize to ensure that the output is valid probability distribution        
        posterior_belief = (belief * measurement_likelihood)
        normalized_posterior_belief = posterior_belief / np.sum(posterior_belief)

    return (measurement_likelihood, normalized_posterior_belief)

def getSegmentDistance(segment):
    x_c = (segment.points[0].x + segment.points[1].x) / 2
    y_c = (segment.points[0].y + segment.points[1].y) / 2
    return sqrt(x_c**2 + y_c**2)