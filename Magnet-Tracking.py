###################### IMPORTS ######################
from copy import deepcopy
from IPython.display import display
from itertools import permutations, combinations
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import art3d
from multiprocess import Array, Process, Queue, Value
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.signal import cont2discrete
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
from traceback import extract_tb
import csv
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import os
import re
import scipy.linalg as la
import sys
import tempfile
import time
import traceback
import warnings

###################### CONSTANTS ######################
NUM_PARAMS = 6
NUM_COMPS = 3
G_FIELD = np.array([-20, 20, -80])*1e0                  # µT
GEO_SCALE = 1e6                                         # µT/T
MAX_GEO = 60e0                                          # µT
NOISE_STD = 2.5                                         # µT
DT = 1e-3                   # s
MAG_STRENGTH = 1.39         # nT*m^3 = T*mm^3
MAG_RADIUS = 1.5            # mm
SENSOR_WIDTH = 2.5          # mm
EPS = 1e0                   # degrees
AREA_FRACTION = 0.5
MIN_DIST = 5e0              # mm
MAX_DIST = 80e0             # mm
MIN_Z = 20e0                # mm
MAX_Z = 40e0                # mm
CHANGE = ['X', 'Y', 'Z']
SPACES = f"{' ' * 100}\n"

class MagnetTrackingSystem:
###################### INITIALIZATION ######################
    def __init__(self, file=None, dist=None, scalar_m=MAG_STRENGTH, radius=MAG_RADIUS, is_euclid=True, scale_down=False, num_magnets=2, num_boards=1, width=2, length=2, height=1, spacing=SENSOR_WIDTH, center=None, angles=None, sys_rot=(0, 0), pos_tau=1, mom_tau=1, geo_tau=1, dt=DT, pos_std=2, mom_std=0.45, geo_std=NOISE_STD, noise_std=NOISE_STD, sensor_noise_std=NOISE_STD):
        self.scalar_m = float(scalar_m)
        self.radius = float(radius)
        self.is_euclid = is_euclid
        self.num_params = NUM_PARAMS - (not self.is_euclid)
        self.num_magnets = num_magnets
        self.num_boards = num_boards
        self.rot = sys_rot if type(sys_rot) == Rot else Rot.from_euler('yz', sys_rot, degrees=True) if len(sys_rot) == 2 else Rot.from_euler('xyz', sys_rot, degrees=True)
        self.pos_tau = float(pos_tau)
        self.mom_tau = float(mom_tau)
        self.geo_tau = float(geo_tau)
        self.dt = float(dt)
        self.pos_std = float(pos_std) if not hasattr(pos_std, '__len__') else type(pos_std)(float(num) for num in pos_std)
        self.mom_std = float(mom_std) if not hasattr(mom_std, '__len__') else type(mom_std)(float(num) for num in mom_std)
        self.geo_std = float(geo_std) if not hasattr(geo_std, '__len__') else type(geo_std)(float(num) for num in geo_std)
        self.noise_std = float(noise_std)
        self.sensor_noise_std = float(sensor_noise_std)
        self.G_FIELD = G_FIELD
        self.scale_down = scale_down
        
        # Bounds for the sensor system
        self.x_range, self.y_range, self.z_range = np.empty((self.num_boards, 2)), np.empty((self.num_boards, 2)), np.empty((self.num_boards, 2))
        # Initialize the sensor system
        self.generate_sensor_system(width, length, height, spacing, center, angles)
        if file is not None:
            self.true_params = None
            self.read_csv(file, width * length * height)
            if dist is not None:
                self.add_distance(dist)
        else:
            self.from_file = False
            self.active_sensors_list = np.arange(self.total_num_sensors)
            self.active_sensors_indices = deepcopy(self.sensor_indices)
            self.x_range = np.array([[sensor_board.x_min, sensor_board.x_max] for sensor_board in self.sensor_boards])
            self.y_range = np.array([[sensor_board.y_min, sensor_board.y_max] for sensor_board in self.sensor_boards])
            self.z_range = np.array([[sensor_board.z_min, sensor_board.z_max] for sensor_board in self.sensor_boards])
            self.distance = None
        self.active_sensors_global_positions = np.concatenate([sensor_board.global_positions[self.active_sensors_indices[i]] for i, sensor_board in enumerate(self.sensor_boards)])
        
        if not self.from_file:
            # Initialize the sensor data, noise, and true parameters
            self.compute_sensor_data()
        # Initialize the covariance matrices
        self.cov_matrices()
        
    # Return the object's parameters
    def __str__(self):
        s = ""
        if self.from_file and hasattr(self, 'file'):
            s += f"Data from file: {self.file}\n"
        s += f"Scalar Magnetization: {self.scalar_m} nT*m^3\n"
        s += f"Number of Magnets: {self.num_magnets}\n"
        s += f"Number of Sensor Boards: {self.num_boards}\n"
        s += f"Number of Active Senors per Board: {self.num_sensors}\n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s += f"System Rotation: {np.array2string(self.rot.as_euler('xyz', degrees=True), separator=', ', precision=NUM_COMPS)}\n"
        s += f"Pos Tau: {round(self.pos_tau, NUM_COMPS)} s, Moment Tau: {round(self.mom_tau, NUM_COMPS)} s, Geo Tau: {round(self.geo_tau, NUM_COMPS)} s, Time Step: {round(self.dt, NUM_COMPS)} s\n"
        s += f"Position STD: {round(self.pos_std, NUM_COMPS) if not hasattr(self.pos_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.pos_std)} mm, Moment STD: {round(self.mom_std, NUM_COMPS) if not hasattr(self.mom_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.mom_std)}{'' if self.is_euclid else ' degrees'}, Geo STD: {round(self.geo_std, NUM_COMPS)  if not hasattr(self.geo_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.geo_std)} µT, Noise STD: {round(self.noise_std, NUM_COMPS)} µT, Sensor Noise STD: {round(self.sensor_noise_std, NUM_COMPS)} µT\n"
        s += f"\n----- Sensor Board Parameters: -----\n"
        for board_idx, sensor_board in enumerate(self.sensor_boards):
            s += f"\nSensor Board {board_idx}:\n"
            s += sensor_board.print_params()
            s += str(sensor_board)
        return s
        
    # Read the CSV file and store the data
    def read_csv(self, file, num_sensors):
        self.from_file = True
        self.file = file
        print(f"Reading data from {file}...")
        data = np.genfromtxt(file, delimiter=',', names=True)
        names = data.dtype.names
        self.data = np.empty(data.shape[0], dtype=object)
        # Collects sensor names
        Bs = [name for name in names if name.startswith('B_')]
        B_raws = [name for name in names if name.startswith('RawB_')]
        Brecons = [name for name in names if name.startswith('Brecon_')]
        # Collects used sensor indices
        if len(Bs) != num_sensors * self.num_boards:
            white_list = np.array([int(Bs[index].split('_')[-1]) for index in range(0, len(Bs), NUM_COMPS)])
            self.sensor_indices = []
            for i in range(self.num_boards):
                self.sensor_indices.append(white_list[(white_list >= i * num_sensors) & (white_list < (i+1) * num_sensors)] % num_sensors)
            self.num_sensors = [len(self.sensor_indices[i]) for i in range(self.num_boards)]
            self.total_num_sensors = int(np.sum(self.num_sensors))
        self.active_sensors_list = np.arange(self.total_num_sensors)
        self.active_sensors_indices = deepcopy(self.sensor_indices)
        # Set bounds for the sensor system
        for i in range(self.num_boards):
            self.x_range[i, 0], self.y_range[i, 0], self.z_range[i, 0] = np.min(self.sensor_boards[i].local_positions[self.active_sensors_indices[i]], axis=0)
            self.x_range[i, 1], self.y_range[i, 1], self.z_range[i, 1] = np.max(self.sensor_boards[i].local_positions[self.active_sensors_indices[i]], axis=0)
        
        # Store the data
        with tqdm(total=data.shape[0]) as pbar:
            for i in range(data.shape[0]):
                # Update the progress bar
                time.sleep(0.01)
                pbar.update(1)
                # Initialize the dictionary for each iteration
                self.data[i] = {}
                self.data[i]['B'] = np.empty((len(Bs)//NUM_COMPS, NUM_COMPS))
                if len(B_raws) > 0:
                    self.data[i]['B_raw'] = np.empty((len(B_raws)//NUM_COMPS, NUM_COMPS))
                if len(Brecons) > 0:
                    self.data[i]['B_recon'] = np.empty((len(Brecons)//NUM_COMPS, NUM_COMPS))
                
                # Store the magnetic field data
                for j in range(0, len(Bs), NUM_COMPS):
                    self.data[i]['B'][j//NUM_COMPS] = np.array([data[i][Bs[j]], data[i][Bs[j+1]], data[i][Bs[j+2]]]) * GEO_SCALE                            # T -> µT
                    if len(B_raws) > 0:
                        self.data[i]['B_raw'][j//NUM_COMPS] = np.array([data[i][B_raws[j]], data[i][B_raws[j+1]], data[i][B_raws[j+2]]]) * GEO_SCALE        # T -> µT
                        self.data[i]['B_recon'][j//NUM_COMPS] = np.array([data[i][Brecons[j]], data[i][Brecons[j+1]], data[i][Brecons[j+2]]]) * GEO_SCALE   # T -> µT
                # Store the system time
                self.data[i]['System Time'] = (data[i]['System_Ts_us'] - data[0]['System_Ts_us']) / GEO_SCALE        # µs -> s
                # Store the magnet parameters and other data
                if 'x0' in names:
                    self.data[i]['Scalar M'] = data[i]['m0'] * 1e9 if data[i]['m0'] != 0 else MAG_STRENGTH           # T*m^3 -> nT*m^3
                    self.data[i]['Magnet Params'] = np.empty((self.num_magnets, NUM_PARAMS-1))
                    for j in range(self.num_magnets):
                        self.data[i]['Magnet Params'][j] = np.array([data[i][f'x{j}'], data[i][f'y{j}'], data[i][f'z{j}'], np.rad2deg(data[i][f'theta{j}']), np.rad2deg(data[i][f'phi{j}'])])
                    if self.is_euclid:
                        self.data[i]['Magnet Params'] = np.hstack((self.data[i]['Magnet Params'][:, :NUM_COMPS], self.get_moment(self.data[i]['Magnet Params'][:, NUM_COMPS:])))
                    self.data[i]['Magnet Params'][:, :NUM_COMPS] *= 1e3                                              # m -> mm
                    self.data[i]['G'] = np.array([data[i]['Gx'], data[i]['Gy'], data[i]['Gz']]) * GEO_SCALE          # T -> µT
                    if (self.data[i]['G'] == np.zeros(NUM_COMPS)).all():
                        self.data[i]['G'] = None
                    if 'd' in names:
                        self.data[i]['Distance'] = data[i]['d'] * 1e3                                                # m -> mm
                    if 'Gyro_X' in names:
                        self.data[i]['Gyro'] = np.array([data[i]['Gyro_X'], data[i]['Gyro_Y'], data[i]['Gyro_Z']])
                        delta_t = self.data[i]['System Time'] - self.data[i-1]['System Time'] if i > 0 else self.dt
                        self.data[i]['Omega'] = Rot.from_euler('xyz', self.data[i]['Gyro']*delta_t, degrees=True).as_matrix()
                    if 'Accel_X' in names:
                        self.data[i]['Accel'] = np.array([data[i]['Accel_X'], data[i]['Accel_Y'], data[i]['Accel_Z']])
        self.update_data(0)
       
    # Generate the sensor system
    def generate_sensor_system(self, grid_width=2, grid_length=2, grid_height=1, spacing=SENSOR_WIDTH, center=None, angles=None):
        self.sensor_indices = [np.arange(grid_width * grid_length * grid_height) for _ in range(self.num_boards)]
        self.num_sensors = [len(self.sensor_indices[i]) for i in range(self.num_boards)]
        self.total_num_sensors = int(np.sum(self.num_sensors))
        self.sensor_boards = []
        # Adjust center and angles and check that they're valid
        center = [(0, 0, 0)] * self.num_boards if center is None else [center] if not hasattr(center[0], "__len__") else center
        angles = [(0, 0)] * self.num_boards if angles is None else [angles] if not hasattr(angles[0], "__len__") else angles
        assert(len(center) == self.num_boards - 1 or len(center) == self.num_boards)
        assert(len(angles) == self.num_boards - 1 or len(angles) == self.num_boards)
        self.sensor_boards.append(Sensor(grid_width, grid_length, grid_height, spacing, center[0] if len(center) == self.num_boards else (0, 0, 0), angles[0] if len(angles) == self.num_boards else (0, 0)))
        if self.num_boards > 1:
            for c, a in zip(center[1-self.num_boards:], angles[1-self.num_boards:]):
                self.sensor_boards.append(Sensor(grid_width, grid_length, grid_height, spacing, c, a))
    
    # Define the true magnet parameters and compute the sensor data
    def compute_sensor_data(self, true_params=None, rotation=False):
        self.from_file = False
        if true_params is not None:
            true_params = true_params[:self.num_magnets, :self.num_params]
        if true_params is None or true_params.size < self.num_magnets * self.num_params:
            true_params = np.concatenate((np.empty(0) if true_params is None else true_params.ravel(), self.random_initial_guess(num_magnets=self.num_magnets if true_params is None else self.num_magnets - true_params.size // self.num_params, rotation=rotation))).reshape(-1, self.num_params)
        self.true_params = true_params.astype(np.float64)
        self.B_sensor = self.evaluate_field(true_params.ravel())

    # Initializes covariance matrices
    def cov_matrices(self):
        N_POS = 2 * NUM_COMPS * self.num_magnets
        N_MOM = 2 * (self.num_params - NUM_COMPS) * self.num_magnets
        N_GEO = 2 * NUM_COMPS
        N_POS_GEO = N_POS + N_GEO
        N_MOM_GEO = N_MOM + N_GEO
        N_NO_GEO = N_POS + N_MOM
        N = N_NO_GEO + N_GEO
        M = self.total_num_sensors * NUM_COMPS
        
        # Indices for state transition and control input matrices
        pos_geo_idx = np.arange(NUM_PARAMS) + np.repeat([N_POS//2, N_POS], NUM_COMPS)
        mom_geo_idx = np.arange(NUM_PARAMS) + np.repeat([N_MOM//2, N_MOM], NUM_COMPS)
        no_geo_idx = np.arange(NUM_PARAMS) + np.repeat([N_NO_GEO//2, N_NO_GEO], NUM_COMPS)
        mom_idx = np.arange(N_MOM) + np.repeat(NUM_COMPS*(np.arange(2 * self.num_magnets) + 1), self.num_params - NUM_COMPS)
        
        # Initialize the state transition matrix
        F_POS = (-np.eye(N_POS) + np.eye(N_POS, k=N_POS//2)) / self.pos_tau
        F_MOM = (-np.eye(N_MOM) + np.eye(N_MOM, k=N_MOM//2)) / self.mom_tau
        F_GEO = (-np.eye(N_GEO) + np.eye(N_GEO, k=N_GEO//2)) / self.geo_tau
        F_POS_GEO = (-np.eye(N_POS_GEO) + np.eye(N_POS_GEO, k=N_POS_GEO//2)) / self.pos_tau
        F_POS_GEO[np.ix_(pos_geo_idx, pos_geo_idx)] *= self.pos_tau / self.geo_tau        
        F_MOM_GEO = (-np.eye(N_MOM_GEO) + np.eye(N_MOM_GEO, k=N_MOM_GEO//2)) / self.mom_tau
        F_MOM_GEO[np.ix_(mom_geo_idx, mom_geo_idx)] *= self.mom_tau / self.geo_tau
        F_NO_GEO = (-np.eye(N_NO_GEO) + np.eye(N_NO_GEO, k=N_NO_GEO//2)) / self.pos_tau
        F_NO_GEO[np.ix_(mom_idx, mom_idx)] *= self.pos_tau / self.mom_tau
        F = (-np.eye(N) + np.eye(N, k=N//2)) / self.pos_tau
        F[np.ix_(mom_idx + np.repeat([0, NUM_COMPS], N_MOM//2), mom_idx + np.repeat([0, NUM_COMPS], N_MOM//2))] *= self.pos_tau / self.mom_tau
        F[np.ix_(no_geo_idx, no_geo_idx)] *= self.pos_tau / self.geo_tau
        # Initialize the control input matrix
        G_POS = np.eye(N_POS, N_POS//2, -N_POS//2) / self.pos_tau
        G_MOM = np.eye(N_MOM, N_MOM//2, -N_MOM//2) / self.mom_tau
        G_GEO = np.eye(N_GEO, N_GEO//2, -N_GEO//2) / self.geo_tau
        G_POS_GEO = np.eye(N_POS_GEO, N_POS_GEO//2, -N_POS_GEO//2) / self.pos_tau
        G_POS_GEO[-NUM_COMPS:, -NUM_COMPS:] *= self.pos_tau / self.geo_tau
        G_MOM_GEO = np.eye(N_MOM_GEO, N_MOM_GEO//2, -N_MOM_GEO//2) / self.mom_tau
        G_MOM_GEO[-NUM_COMPS:, -NUM_COMPS:] *= self.mom_tau / self.geo_tau
        G_NO_GEO = np.eye(N_NO_GEO, N_NO_GEO//2, -N_NO_GEO//2) / self.pos_tau
        G_NO_GEO[np.ix_(mom_idx[-N_MOM//2:], mom_idx[:N_MOM//2])] *= self.pos_tau / self.mom_tau
        G = np.eye(N, N//2, -N//2) / self.pos_tau
        G[np.ix_(mom_idx[-N_MOM//2:] + NUM_COMPS, mom_idx[:N_MOM//2])] *= self.pos_tau / self.mom_tau
        G[-NUM_COMPS:, -NUM_COMPS:] *= self.pos_tau / self.geo_tau
        # Discretize the state transition and control input matrices
        F_POS, G_POS = cont2discrete((F_POS, G_POS, None, None), dt=self.dt)[:2]
        F_MOM, G_MOM = cont2discrete((F_MOM, G_MOM, None, None), dt=self.dt)[:2]
        F_GEO, G_GEO = cont2discrete((F_GEO, G_GEO, None, None), dt=self.dt)[:2]
        F_POS_GEO, G_POS_GEO = cont2discrete((F_POS_GEO, G_POS_GEO, None, None), dt=self.dt)[:2]
        F_MOM_GEO, G_MOM_GEO = cont2discrete((F_MOM_GEO, G_MOM_GEO, None, None), dt=self.dt)[:2]
        F_NO_GEO, G_NO_GEO = cont2discrete((F_NO_GEO, G_NO_GEO, None, None), dt=self.dt)[:2]
        F, G = cont2discrete((F, G, None, None), dt=self.dt)[:2]
        
        # Initialize the position, moment, and geo standard deviations
        # Position
        if not hasattr(self.pos_std, "__len__"):
            x_std = y_std = z_std = self.pos_std
        elif len(self.pos_std) == 2:
            x_std = y_std = self.pos_std[0]
            z_std = self.pos_std[1]
        elif len(self.pos_std) == 3:
            x_std, y_std, z_std = self.pos_std
        else:
            raise ValueError("Position standard deviation must be a scalar, pair, or triplet")
        # Moment
        if not hasattr(self.mom_std, "__len__"):
            mx_std = my_std = mz_std = theta_std = phi_std = self.mom_std
        elif len(self.mom_std) == 2:
            mx_std = my_std = theta_std = self.mom_std[0]
            mz_std = phi_std = self.mom_std[1]
        elif len(self.mom_std) == 3 and self.is_euclid:
            mx_std, my_std, mz_std = self.mom_std
        else:
            raise ValueError("Moment standard deviation must be a scalar or pair")
        # Geo
        if not hasattr(self.geo_std, "__len__"):
            geo_x_std = geo_y_std = geo_z_std = self.geo_std
        elif len(self.geo_std) == 2:
            geo_x_std = geo_y_std = self.geo_std[0]
            geo_z_std = self.geo_std[1]
        elif len(self.geo_std) == 3:
            geo_x_std, geo_y_std, geo_z_std = self.geo_std
        else:
            raise ValueError("Geo standard deviation must be a scalar, pair, or triplet")
        geo_std = np.array([geo_x_std, geo_y_std, geo_z_std]) / (1 if self.scale_down else GEO_SCALE)
        
        P_POS = np.eye(N_POS)
        P_MOM = np.eye(N_MOM)
        P_GEO = np.eye(N_GEO)
        P_POS_GEO = np.eye(N_POS_GEO)
        P_MOM_GEO = np.eye(N_MOM_GEO)
        P_NO_GEO = np.eye(N_NO_GEO)
        P = np.eye(N)
        
        if self.is_euclid:
            Q_diag = np.tile([x_std, y_std, z_std, mx_std, my_std, mz_std], self.num_magnets)
            Q_mom_diag = np.tile([mx_std, my_std, mz_std], self.num_magnets)
        else:
            Q_diag = np.tile([x_std, y_std, z_std, theta_std, phi_std], self.num_magnets)
            Q_mom_diag = np.tile([theta_std, phi_std], self.num_magnets)
        Q_pos_diag = np.tile([x_std, y_std, z_std], self.num_magnets)
        
        Q_POS = np.diag(Q_pos_diag)**2
        Q_MOM = np.diag(Q_mom_diag)**2
        Q_GEO = np.diag(geo_std)**2
        Q_POS_GEO = np.diag(np.concatenate((Q_pos_diag, geo_std)))**2
        Q_MOM_GEO = np.diag(np.concatenate((Q_mom_diag, geo_std)))**2
        Q_NO_GEO = np.diag(Q_diag)**2
        Q = np.diag(np.concatenate((Q_diag, geo_std)))**2
        # Initialized P at steady state
        for _ in range(int(5e4)):
            P_POS = F_POS.dot(P_POS).dot(F_POS.T) + G_POS.dot(Q_POS).dot(G_POS.T)
            P_MOM = F_MOM.dot(P_MOM).dot(F_MOM.T) + G_MOM.dot(Q_MOM).dot(G_MOM.T)
            P_GEO = F_GEO.dot(P_GEO).dot(F_GEO.T) + G_GEO.dot(Q_GEO).dot(G_GEO.T)
            P_POS_GEO = F_POS_GEO.dot(P_POS_GEO).dot(F_POS_GEO.T) + G_POS_GEO.dot(Q_POS_GEO).dot(G_POS_GEO.T)
            P_MOM_GEO = F_MOM_GEO.dot(P_MOM_GEO).dot(F_MOM_GEO.T) + G_MOM_GEO.dot(Q_MOM_GEO).dot(G_MOM_GEO.T)
            P_NO_GEO = F_NO_GEO.dot(P_NO_GEO).dot(F_NO_GEO.T) + G_NO_GEO.dot(Q_NO_GEO).dot(G_NO_GEO.T)
            P = F.dot(P).dot(F.T) + G.dot(Q).dot(G.T)
        # State transition matrices
        self.F_POS = F_POS
        self.F_MOM = F_MOM
        self.F_GEO = F_GEO
        self.F_POS_GEO = F_POS_GEO
        self.F_MOM_GEO = F_MOM_GEO
        self.F_NO_GEO = F_NO_GEO
        self.F = F
        # Control input matrices
        self.G_POS = G_POS
        self.G_MOM = G_MOM
        self.G_GEO = G_GEO
        self.G_POS_GEO = G_POS_GEO
        self.G_MOM_GEO = G_MOM_GEO
        self.G_NO_GEO = G_NO_GEO
        self.G = G
        # State estimate covariance matrices
        self.P0_POS = P_POS
        self.P0_MOM = P_MOM
        self.P0_GEO = P_GEO
        self.P0_POS_GEO = P_POS_GEO
        self.P0_MOM_GEO = P_MOM_GEO
        self.P0_NO_GEO = P_NO_GEO
        self.P0 = P
        # Process noise covariance matrices
        self.Q_POS = Q_POS
        self.Q_MOM = Q_MOM
        self.Q_GEO = Q_GEO
        self.Q_POS_GEO = Q_POS_GEO
        self.Q_MOM_GEO = Q_MOM_GEO
        self.Q_NO_GEO = Q_NO_GEO
        self.Q = Q
        # Measurement noise covariance matrix
        self.R = np.eye(M) * (self.noise_std / GEO_SCALE)**2
 
###################### CONVERSION FUNCTIONS ######################
    # Converts magnet parameters with angles to magnet parameters with moments
    def set_with_angles(self, true_params, ret=True, num_magnets=None):
        num_magnets = self.num_magnets if num_magnets is None else num_magnets
        true_params = true_params.reshape(num_magnets, -1)
        params = (
            np.hstack((true_params[:, :NUM_COMPS], self.get_moment(true_params[:, NUM_COMPS:]))) if self.is_euclid else
            deepcopy(true_params)
        )
        if ret:
            return params
        else:
            self.compute_sensor_data(params if self.is_euclid else true_params)
    
    # Sets magnet parameters with random moments and given positions
    def set_with_pos(self, true_params, ret=True):
        true_params = true_params.reshape(self.num_magnets, -1)
        params = np.zeros((self.num_magnets, NUM_PARAMS))
        for i in range(self.num_magnets):
            pos = true_params[i]
            moment = npr.rand(NUM_COMPS) * 2 - 1
            moment /= la.norm(moment)
            params[i] = np.concatenate([pos, moment])
        if not self.is_euclid:
            params = np.hstack((params[:, :NUM_COMPS], self.get_angles(params[:, NUM_COMPS:])))
        if ret:
            return params
        else:
            self.compute_sensor_data(params)
    
    # Converts magnet parameters to moment
    def get_moment(self, angles):
        return Rot.from_euler('yz', angles, degrees=True).apply([0, 0, 1])
    
    # Converts magnet parameters to angles
    def get_angles(self, moments):
        normalized_moments = moments / la.norm(moments, axis=-1).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angles = np.flip([Rot.align_vectors(moment, [0, 0, 1])[0].as_euler('ZYZ', degrees=True) for moment in normalized_moments], axis=-1)[:, -2:]
        return angles if moments.ndim > 1 else angles[0]

###################### OPTIMIZATION HELPER FUNCTIONS ######################
    # Evaluate the magnetic field expression
    def evaluate_field(self, magnet_params):
        if self.num_magnets > 0:
            magnets = deepcopy(magnet_params).ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params)
            moment_vecs = self.scalar_m * (magnets[:, NUM_COMPS:] if self.is_euclid else self.get_moment(magnets[:, NUM_COMPS:]))                                       # num magnets x 3
            pos_matrix = self.active_sensors_global_positions[:, None] - magnets[:, :NUM_COMPS][None]                                                                   # num sensors x num magnets x 3
            # Compute pos_norm and set to 1 where pos_norm == 0 to avoid division by zero
            pos_norm = la.norm(pos_matrix, axis=-1)[..., None]
            indices = (pos_norm == 0)[..., 0]
            pos_norm[indices] = 1
            pos_hat = pos_matrix / pos_norm
            # Compute the magnetic field and set to inf where pos_norm == 0
            B = (3 * np.vecdot(pos_hat, moment_vecs)[..., None] * pos_hat - moment_vecs) / pos_norm**3
            B[indices] = np.inf
            # Sum the magnetic field contributions from all magnets
            return np.sum(B, axis=1)
        else:
            return np.zeros(self.B_sensor.shape)
    
    # Computes cost of magnetic field
    def cost_function(self, magnet_params, method='eif', with_geo=True, with_noise=False):
        # Add Gaussian noise/geomagnetic disturbance if needed
        noise = np.zeros(self.B_sensor.shape)
        if with_noise:
            noise += npr.normal(0, self.sensor_noise_std, self.B_sensor.shape)
        if with_geo and not self.from_file:
            noise += self.G_FIELD
        # Compute the magnetic field
        B_model = (
            (self.evaluate_field(magnet_params) + magnet_params[-NUM_COMPS:].reshape(1, -1) * with_geo / (GEO_SCALE if self.scale_down else 1))
        )
        B_sensor = self.B_sensor + noise / GEO_SCALE
        B_cost = (B_model - B_sensor).ravel()
        return B_cost

    # Evaluate the Jacobian expression for each sensor
    def jacobian_function(self, magnet_params, method='eif', with_geo=False, with_noise=False, fixed_pos=False, fixed_moment=False):
        num_params = (not fixed_pos)*NUM_COMPS + (not fixed_moment)*(NUM_COMPS - (not self.is_euclid))
        jac = np.zeros((self.total_num_sensors * NUM_COMPS, num_params * self.num_magnets + NUM_COMPS * with_geo))
        # Process all sensors at once
        if self.num_magnets > 0:
            magnets = magnet_params.ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params)
            magnet_offset = np.arange(self.num_magnets).reshape(1, -1)*num_params
            # Pre-compute moment vectors and derivatives for all magnets
            moment_vecs = self.scalar_m * (magnets[:, NUM_COMPS:] if self.is_euclid else self.get_moment(magnets[:, NUM_COMPS:]))               # num magnets x 3
            if not self.is_euclid:
                thetas, phis = magnets[:, NUM_COMPS:].T
                deriv_moment_vecs = self.scalar_m * np.array([[np.cos(np.deg2rad(thetas))*np.cos(np.deg2rad(phis)), np.cos(np.deg2rad(thetas))*np.sin(np.deg2rad(phis)), -np.sin(np.deg2rad(thetas))],
                                                            [-np.sin(np.deg2rad(thetas))*np.sin(np.deg2rad(phis)), np.sin(np.deg2rad(thetas))*np.cos(np.deg2rad(phis)), np.zeros_like(thetas)]]).transpose(-1, 0, 1)
            
            sensor_offset = np.arange(self.total_num_sensors).reshape(-1, 1)*NUM_COMPS
            pos_matrix = self.active_sensors_global_positions[:, None] - magnets[:, :NUM_COMPS][None]                                           # num sensors x num magnets x 3
            pos_norm = la.norm(pos_matrix, axis=-1)
            pos_norm_3 = pos_norm**3
            pos_norm_5 = pos_norm**5
            pos_norm_7 = pos_norm**7
            m_dot_p = np.vecdot(moment_vecs, pos_matrix)
            if not self.is_euclid:
                dm_dot_p = np.array([np.vecdot(deriv_moment_vecs[:, 0], pos_matrix), np.vecdot(deriv_moment_vecs[:, 1], pos_matrix)])

            for a in range(NUM_COMPS):
                for b in range(NUM_COMPS*fixed_pos, NUM_COMPS if fixed_moment else self.num_params):
                    jac[sensor_offset+a, magnet_offset+b] = (
                        (
                            ((- 3 * m_dot_p / pos_norm_5) if a == b else 0)
                            - 3 * moment_vecs[..., a] * pos_matrix[..., b] / pos_norm_5 
                            - 3 * moment_vecs[..., b] * pos_matrix[..., a] / pos_norm_5 
                            + 15 * pos_matrix[..., a] * pos_matrix[..., b] * m_dot_p / pos_norm_7
                        ) if b < NUM_COMPS else (                                                       # Position derivatives (∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z, ∂By/∂x, ∂By/∂y, ∂By/∂z, ∂Bz/∂x, ∂Bz/∂y, ∂Bz/∂z)
                            ((- 1 / pos_norm_3) if a == b - NUM_COMPS else 0)
                            + 3 * pos_matrix[..., a] * pos_matrix[..., b - NUM_COMPS] / pos_norm_5
                        ) if self.is_euclid and b >= NUM_COMPS else (                                   # Euclidean moment derivatives (∂Bx/∂mx, ∂Bx/∂my, ∂Bx/∂mz, ∂By/∂mx, ∂By/∂my, ∂By/∂mz, ∂Bz/∂mx, ∂Bz/∂my, ∂Bz/∂mz)
                            3 * pos_matrix[..., a] * dm_dot_p[b - NUM_COMPS] / pos_norm_5
                            - deriv_moment_vecs[:, b - NUM_COMPS, a] / pos_norm_3
                        )                                                                               # Spherical moment derivatives (∂Bx/∂θ, ∂Bx/∂Φ, ∂By/∂yθ, ∂By/∂yΦ, ∂Bz/∂zθ, ∂Bz/∂zΦ)
                    )
        # Add disturbance field compensation if needed
        if with_geo:
            jac[:, -NUM_COMPS:] = np.tile(np.eye(NUM_COMPS), self.total_num_sensors).T / (GEO_SCALE if self.scale_down else 1)
        return jac

    # Computes the Jacobian, then add zeros to the right side to account for the complementary state
    def kalman_jacobian_function(self, magnet_params, with_geo=False, fixed_pos=False, fixed_moment=False):
        jacobian_cost = self.jacobian_function(magnet_params, 'eif', with_geo, fixed_pos=fixed_pos, fixed_moment=fixed_moment)
        return np.concatenate([jacobian_cost, np.zeros(jacobian_cost.shape)], axis=1)
    
###################### MAGNET PARAMETER FUNCTIONS ######################
    # Transforms the magnet parameters to the sensor board frame
    def transform_magnet_to_board_frame(self, magnet_params, sensor_board, global_local=True):
        num_magnets = len(magnet_params.ravel()) // self.num_params
        transformed_magnets = deepcopy(magnet_params).ravel()[:self.num_params*num_magnets].reshape(num_magnets, -1)
        
        sensor_board_rot = sensor_board.rot.inv() if global_local else sensor_board.rot
        sensor_board_center = np.array(sensor_board.center)
        transformed_magnets[:, :NUM_COMPS] = np.round(sensor_board_rot.apply(transformed_magnets[:, :NUM_COMPS] - sensor_board_center * global_local) + sensor_board_center * (not global_local), NUM_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_magnets[:, NUM_COMPS:] = np.round((
                sensor_board_rot.apply(transformed_magnets[:, NUM_COMPS:]) if self.is_euclid else
                np.flip((sensor_board_rot * Rot.from_euler('yz', transformed_magnets[:, NUM_COMPS:], degrees=True)).as_euler('ZYZ', degrees=True).reshape(-1, NUM_COMPS)[:, :2], -1)
            ), NUM_PARAMS)
        
        return transformed_magnets
    
    # Replaces a magnet if it is too close to another magnet
    def replace_magnet(self, magnet_params, index):
        too_close = True
        while too_close:
            rand_board_idx = npr.randint(self.num_boards)
            rand_board = self.sensor_boards[rand_board_idx]
            pos = np.round(np.array([npr.uniform(self.x_range[rand_board_idx, 0], self.x_range[rand_board_idx, 1]) * AREA_FRACTION, npr.uniform(self.y_range[rand_board_idx, 0], self.y_range[rand_board_idx, 1]) * AREA_FRACTION, npr.uniform(self.z_range[rand_board_idx, 1] + MIN_Z, self.z_range[rand_board_idx, 1] + MAX_Z)]), NUM_COMPS)
            pos = self.transform_magnet_to_board_frame(np.array([pos, [0, 0, 1]]).ravel(), rand_board)[0, :NUM_COMPS]
            too_close = False
            for i in range(len(magnet_params)):
                if i != index and (np.abs(magnet_params[i, :NUM_COMPS] - pos) < MIN_DIST).any():
                    too_close = True
                    break
        moment = npr.rand(NUM_COMPS) * 2 - 1
        moment /= la.norm(moment)
        return np.concatenate([pos, moment if self.is_euclid else self.get_angles(moment)])
    
    # Adjusts the magnet parameters if they are too far from the sensor system
    def reset_guess(self, magnet_params, fixed_moment=False):
        # Check if there are any magnets
        if self.num_magnets == 0:
            return magnet_params.ravel(), False
        
        params = deepcopy(magnet_params[:self.num_magnets * self.num_params]).reshape(-1, self.num_params)
        changed_params = False
        
        # Ensure moments are normalized
        params = self.normalize_moments(params.ravel(), normalize=True).reshape(-1, self.num_params)
        
        # Check if each magnet is within at least one sensor board
        for i, magnet in enumerate(params):
            in_bounds = False
            for board_idx, sensor_board in enumerate(self.sensor_boards):
                transform_magnet = self.transform_magnet_to_board_frame(magnet, sensor_board)[0]
                if transform_magnet[2] >= self.z_range[board_idx, 1] and (transform_magnet[2] < self.z_range[board_idx, 1] + 2*MAX_Z or np.min(la.norm(transform_magnet[:NUM_COMPS] - sensor_board.local_positions, axis=1)) < 2*MAX_Z):
                    in_bounds = True
                    break
            if not in_bounds:
                params[i] = self.replace_magnet(params, i)
                changed_params = True
       
        # Check if the magnet parameters are too close to each other
        if in_bounds:
            pairs = combinations(np.arange(self.num_magnets), 2)
            for pair in pairs:
                magnet1, magnet2 = params[pair[0]], params[pair[1]]
                if la.norm(magnet1[:NUM_COMPS] - magnet2[:NUM_COMPS]) < MIN_DIST:
                    params[pair[1]] = self.replace_magnet(params, pair[1])
                    changed_params = True
        if fixed_moment:
            params = params.reshape(-1, self.num_params)
            params[:, NUM_COMPS:] = magnet_params.ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params)[:, NUM_COMPS:]
        # Add the geomagnetic disturbance if needed
        if len(magnet_params.ravel()) > self.num_magnets * self.num_params:
            params = np.concatenate((params.ravel(), magnet_params.ravel()[-NUM_COMPS:]))
        return params, changed_params
    
    # Generates a random initial guess
    def random_initial_guess(self, num_magnets=None, rotation=False, with_geo=False):
        num_magnets = num_magnets if num_magnets is not None else self.num_magnets
        guess = np.zeros((num_magnets, NUM_PARAMS))
        i = 0
        while i < num_magnets:
            valid = True
            board_num = npr.randint(self.num_boards)
            if rotation:
                pos = np.round(Rot.from_euler('ZYX', [npr.uniform(high=2*np.pi), npr.uniform(-np.pi/2, np.pi/2), npr.uniform(-np.pi/2, np.pi/2)]).apply([0, 0, npr.uniform(high=MAX_Z)]), NUM_COMPS)
            else:
                pos = np.round(np.array([npr.uniform(self.x_range[board_num, 0], self.x_range[board_num, 1]) * AREA_FRACTION, npr.uniform(self.y_range[board_num, 0], self.y_range[board_num, 1]) * AREA_FRACTION, npr.uniform(self.z_range[board_num, 1] + MIN_Z, self.z_range[board_num, 1] + MAX_Z)]), NUM_COMPS)
            param = self.transform_magnet_to_board_frame(np.concatenate((pos, [1, 0, 0] if self.is_euclid else [90, 0])), self.sensor_boards[board_num], False)[0]
            pos = param[:NUM_COMPS]
            # Check if magnet is behind any sensor boards
            for board_idx in range(self.num_boards):
                transform_magnet = self.transform_magnet_to_board_frame(param, self.sensor_boards[board_idx])[0, :NUM_COMPS]
                transform_z = transform_magnet[-1]
                if transform_z < self.z_range[board_idx, 1] + 2 * self.radius:
                    valid = False
                    break
            # Check if magnet is too close to another magnet
            if valid:
                for j in range(i):
                    if (np.abs(pos - guess[j, :NUM_COMPS]) < MIN_DIST).any():
                        # ith guess is too close to jth guess, generate new ith guess
                        valid = False
                        break
            if valid:
                moment = npr.rand(NUM_COMPS) * 2 - 1
                mx, my, mz = np.round(moment / la.norm(moment), NUM_COMPS)
                guess[i] = [pos[0], pos[1], pos[2], mx, my, mz]
                i += 1
        # Convert moments to angles if needed  
        if not self.is_euclid and num_magnets > 0:
            guess = np.hstack((guess[:, :NUM_COMPS], np.round(self.get_angles(guess[:, NUM_COMPS:]), NUM_COMPS)))
        # Add the geomagnetic disturbance if needed
        if with_geo:
            vec = npr.rand(NUM_COMPS) * 2 - 1
            vec *= MAX_GEO / la.norm(vec)
            guess = np.concatenate((guess.ravel(), np.round(vec, NUM_COMPS) / (1 if self.scale_down else GEO_SCALE)))
        return guess.ravel()
        
    
###################### LEVENBERG-MARQUARDT OPTIMIZATION ######################
    # Runs the Levenberg-Marquardt optimization
    def run_LM(self, initial_guess=None, result=False, with_geo=False, with_noise=False, max_nfev=None, verbose=2, from_loop=False, iter=0):
        with_geo = with_geo or (self.from_file and 'B_recon' not in self.data[self.iter])
        changed_params = False
        if not from_loop:
            # Set the initial guess if needed
            if initial_guess is None:
                initial_guess = self.random_initial_guess(with_geo=with_geo)
            initial_guess, changed_params = self.reset_guess(initial_guess)
            # Print the optimization settings
            if verbose > 1:
                print(f"Running Levenberg-Marquardt with {'data from file, ' if self.from_file else ''}{'noise, ' if with_noise and not self.from_file else 'no noise, '}{'' if with_geo else 'no '}geo disturbance, {self.num_magnets} magnet(s), and {self.num_boards} board(s)")
                print(f"Initial Guess: \n{np.array2string(initial_guess, separator=', ', precision=NUM_COMPS)}")
            # Extend the guess if needed
            if with_geo:
                vec = npr.rand(NUM_COMPS) * 2 - 1
                vec *= MAX_GEO / la.norm(vec)
                if len(initial_guess.ravel()) == self.num_magnets * self.num_params:
                    initial_guess = np.concatenate((initial_guess.ravel(), np.round(vec, NUM_COMPS) / (1 if self.scale_down else GEO_SCALE)))
                if verbose > 1:
                    print(f"Geo Disturbance (µT): \n{np.array2string(initial_guess[-NUM_COMPS:] * (1 if self.scale_down else GEO_SCALE), separator=', ', precision=NUM_COMPS)}")
            else:
                initial_guess = initial_guess.ravel()[:self.num_magnets * self.num_params]
            # Use reconstructed data if needed
            if self.from_file:
                self.update_data(iter, with_geo)
        # Run the optimization
        results = least_squares(self.cost_function, initial_guess.ravel(), jac=self.jacobian_function, args=('lm', with_geo, with_noise), method='lm', max_nfev=100, verbose=verbose, gtol=max(np.finfo(float).eps, 0), xtol=max(np.finfo(float).eps, 1e-8), ftol=max(np.finfo(float).eps, 1e-14))
        opt_params = results.x
        # Return the results
        if verbose > 0:
            np.set_printoptions(precision=NUM_COMPS)
            print(f"Optimal magnet parameters:\n {np.array2string(opt_params[:self.num_magnets * self.num_params].reshape(-1, self.num_params), separator=', ', precision=NUM_COMPS)}")
            if with_geo:
                print(f"Geo disturbance (µT): {np.array2string(opt_params[-NUM_COMPS:], separator=', ', precision=NUM_COMPS)}")
            print()
            np.set_printoptions()
        if result:
            return opt_params, results, changed_params
        else:
            return opt_params, changed_params
    
    # Runs the Levenberg-Marquardt optimization in a loop
    def run_LM_loop(self, initial_guess=None, iter=5000, with_geo=False, with_noise=False, plotting=False, plot_inter=20, show_dist=False, verbose=2, with_covs=False, moving=False, start_idx=0):
        show_dist = show_dist and self.num_magnets % 2 == 0 and self.num_magnets > 0
        # Set the initial guess if needed
        if initial_guess is None:
            initial_guess = self.random_initial_guess(with_geo=with_geo)
        initial_guess, changed_params = self.reset_guess(initial_guess)
        # Print the optimization settings and set the number of iterations
        moving = moving and self.from_file
        if moving:
            iter = min(iter, len(self.data) - start_idx)
        if self.from_file:
            self.update_data(start_idx, with_geo)
        if verbose == 2:
            print(f"Running Levenberg-Marquardt in loop with with {'data from file, ' if self.from_file else ''}{'noise, ' if with_noise else 'no noise, '}{'' if with_geo else 'no '}geo disturbance, {self.num_magnets} magnet(s), and {self.num_boards} board(s)")
            print(f"Initial Guess: {'***' if changed_params else ''}\n{np.array2string(initial_guess.ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params), separator=', ', precision=NUM_COMPS)}")
        # Extend the guess if needed
        if with_geo:
            vec = npr.rand(NUM_COMPS) * 2 - 1
            vec *= MAX_GEO / la.norm(vec)
            if len(initial_guess.ravel()) == self.num_magnets * self.num_params:
                initial_guess = np.concatenate((initial_guess.ravel(), np.round(vec, NUM_COMPS) / (1 if self.scale_down else GEO_SCALE)))
            if verbose == 2:
                print(f"Geo Disturbance (µT): \n{np.array2string(initial_guess[-NUM_COMPS:] * (1 if self.scale_down else GEO_SCALE), separator=', ', precision=NUM_COMPS)}")
        else:
            initial_guess = initial_guess.ravel()[:self.num_magnets * self.num_params]
        if plotting:
            hdisplay_img1 = display(display_id='State Plot Display')
            hdisplay_img2 = display(display_id='Distance Plot Display')
        if verbose > 0:
            pbar = tqdm(total=iter if not with_covs else iter*MAX_NFEV)
            pbar.clear()
        # Set the number of function evaluations
        N = len(initial_guess)
        MAX_NFEV = 100 * N
        curr_guess = initial_guess.ravel()
        opt_states = np.array([curr_guess])
        final_covs = []
        status = []
        # Run the optimization loop
        try:
            for i in range(iter):
                obs = np.array([curr_guess.ravel()])
                for eval in range(MAX_NFEV if with_covs else 1):
                    # Run LM
                    if with_covs:
                        opt_state = self.run_LM(curr_guess, with_geo=with_geo, with_noise=with_noise, max_nfev=eval+1, verbose=0, from_loop=True)[0]
                        # Update the observations and covariance matrices
                        obs = np.vstack((obs, opt_state.ravel()))
                        curr_cov = np.diag(np.cov(obs.T))
                        covs = curr_cov if eval == 0 else np.vstack((covs, curr_cov))
                    else:
                        opt_state, results = self.run_LM(curr_guess, result=True, with_geo=with_geo, with_noise=with_noise, verbose=0, from_loop=True)[:2]
                        opt_state = self.normalize_moments(opt_state, normalize=True)
                        status.append(results.status)
                    # Update the progress bar
                    if verbose > 0:
                        pbar.update(1)
                        if with_covs:
                            pbar.set_description(f"Iteration: {i+1}/{iter}, eval: {eval+1}")
                        else:
                            pbar.set_description(f"Norm: {la.norm(opt_state.ravel() - curr_guess):.3e}, Cost: {la.norm(self.cost_function(opt_state.ravel(), with_geo=with_geo)):.3e}, {results.cost:.3e}")
                # Update the current guess and store the results
                curr_guess = opt_state.ravel()
                opt_states = np.vstack((opt_states, curr_guess))
                if with_covs:
                    final_covs = np.vstack((final_covs, covs)) if with_covs else final_covs
                if plotting and (i+1) % plot_inter == 0:
                    figs = self.plotting(states=opt_states[len(opt_states)-max(plot_inter, 20)+1:], running=True, moving=moving, show_dist=show_dist, with_true=True, start_idx=start_idx+len(opt_states)-max(plot_inter, 20)-1)
                    hdisplay_img1.update(figs[0])
                    if show_dist:
                        hdisplay_img2.update(figs[1])
                if moving:
                    self.update_data(self.iter + 1, with_geo)
        except KeyboardInterrupt:
            print("\nInterrupted\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            print(traceback.format_exc())
            sys.exit(1)
        finally:
            if verbose > 0:
                pbar.close()
            if plotting:
                plt.close()
            if verbose == 2:
                print(f"\nOptimal magnet parameters:\n {np.array2string(opt_states[-1][:N-NUM_COMPS*(with_geo)].reshape(-1, self.num_params), separator=', ', precision=NUM_COMPS)}")
                if with_geo:
                    print(f"Geo Disturbance (µT): \n{np.array2string(opt_states[-1][-NUM_COMPS:] * (1 if self.scale_down else GEO_SCALE), separator=', ', precision=NUM_COMPS)}")
                print()
            return opt_states, changed_params, status, final_covs if with_covs and moving else covs if with_covs else None
    
###################### EXTENDED FILTERS OPTIMIZATION ######################
    # Initializes the Extended Kalman/Information Filter
    def init_filter(self, initial_guess, lm_guess, iter, init_P, fixed_pos, fixed_moment, fixed_geo, with_geo, with_prints, moving, start_idx):
        initial_guess = deepcopy(initial_guess)
        # Set the initial guess if needed
        if initial_guess is None:
            initial_guess = self.random_initial_guess(with_geo=with_geo)
        if lm_guess is not None:
            orig_lm_guess = lm_guess.ravel()
            lm_guess = lm_guess.ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params)
            avg_dists = np.empty((2*self.num_magnets))
            sensor_pos = np.array([sensor_board.global_positions for sensor_board in self.sensor_boards]).reshape(-1, NUM_COMPS)
            for i in range(self.num_magnets):
                avg_dists[i] = np.mean(la.norm(lm_guess[i, :NUM_COMPS] - sensor_pos, axis=1))
                avg_dists[i + self.num_magnets] = np.mean(la.norm(initial_guess[i, :NUM_COMPS] - sensor_pos, axis=1))
            indices = np.argsort(avg_dists)[:self.num_magnets]
            for i in range(self.num_magnets):
                initial_guess[i] = lm_guess[indices[i]] if indices[i] < self.num_magnets else initial_guess[indices[i] - self.num_magnets]
            # Add the geomagnetic disturbance from the Levenberg-Marquardt guess
            if with_geo and len(orig_lm_guess) > self.num_magnets * self.num_params:
                initial_guess = np.concatenate((initial_guess.ravel(), orig_lm_guess[-NUM_COMPS:]))
        initial_guess, changed_params = self.reset_guess(initial_guess, fixed_moment)
        
        if with_prints:
            print(f"Initial Guess: {'***' if changed_params else ''}\n{np.array2string(initial_guess[:self.num_magnets * self.num_params].reshape(-1, self.num_params), separator=', ', precision=NUM_COMPS)}{'' if with_geo else f'\n'}")
        geo = []
        if fixed_pos or fixed_moment:
            pos, moment = np.hsplit(initial_guess.ravel()[:self.num_magnets * self.num_params].reshape(-1, self.num_params), 2)
            geo = initial_guess.ravel()[self.num_magnets * self.num_params:]
            initial_guess = (
                np.concatenate((moment.ravel(), geo)) if fixed_pos and not fixed_moment else
                np.concatenate((pos.ravel(), geo)) if not fixed_pos and fixed_moment else
                deepcopy(geo) # if fixed_pos and fixed_moment
            )
        if with_geo:
            vec = npr.rand(NUM_COMPS) * 2 - 1
            vec *= MAX_GEO / la.norm(vec)
            if len(initial_guess.ravel()) % self.num_params == 0:
                initial_guess = np.concatenate((initial_guess.ravel(), np.round(vec, NUM_COMPS) / (1 if self.scale_down else GEO_SCALE)))
            if with_prints:
                print(f"Geo Disturbance (µT): \n{np.array2string(initial_guess.ravel()[-NUM_COMPS:] * (1 if self.scale_down else GEO_SCALE), separator=', ', precision=NUM_COMPS)}\n")
            if fixed_geo:
                geo = initial_guess.ravel()[-NUM_COMPS:]
                initial_guess = initial_guess.ravel()[:-NUM_COMPS]
        else:
            initial_guess = initial_guess.ravel()[:self.num_magnets * (self.num_params - NUM_COMPS*fixed_pos - (self.num_params - NUM_COMPS)*fixed_moment)]

        N = len(initial_guess.ravel())
        curr_est = np.concatenate((initial_guess.ravel(), initial_guess.ravel()))
        const_geo = not with_geo or fixed_geo
        F, G, P, Q = deepcopy(
            (self.F, self.G, self.P0, self.Q) if not fixed_pos and not fixed_moment and not const_geo else
            (self.F_NO_GEO, self.G_NO_GEO, self.P0_NO_GEO, self.Q_NO_GEO) if not fixed_pos and not fixed_moment and const_geo else
            (self.F_POS_GEO, self.G_POS_GEO, self.P0_POS_GEO, self.Q_POS_GEO) if not fixed_pos and fixed_moment and not const_geo else
            (self.F_POS, self.G_POS, self.P0_POS, self.Q_POS) if not fixed_pos and fixed_moment and const_geo else
            (self.F_MOM_GEO, self.G_MOM_GEO, self.P0_MOM_GEO, self.Q_MOM_GEO) if fixed_pos and not fixed_moment and not const_geo else
            (self.F_MOM, self.G_MOM, self.P0_MOM, self.Q_MOM) if fixed_pos and not fixed_moment and const_geo else
            (self.F_GEO, self.G_GEO, self.P0_GEO, self.Q_GEO) # if fixed_pos and fixed_moment and not const_geo
        )
        if init_P is not None and not changed_params:
            P = deepcopy(init_P)
        # Check if the data is moving
        if self.from_file:
            self.update_data(start_idx, with_geo)
        moving = moving and self.from_file
        if moving:
            iters_per_iter = np.ceil([(self.data[i+1]['System Time']-self.data[i]['System Time'])/self.dt for i in range(len(self.data)-1)]).astype(int)
            iters_per_iter = np.append(iters_per_iter, int(iters_per_iter.max()))
            iter = min(iter, len(self.data) - start_idx)
        # For plotting/debugging
        state = np.hstack((
            np.hstack((pos, moment)).ravel() if fixed_pos and fixed_moment else
            np.hstack((pos, curr_est[:NUM_COMPS*self.num_magnets].reshape(-1, NUM_COMPS))).ravel() if fixed_pos else
            np.hstack((curr_est[:NUM_COMPS*self.num_magnets].reshape(-1, NUM_COMPS), moment)).ravel() if fixed_moment else
            curr_est[:N-NUM_COMPS*(not const_geo)] # if not fixed_pos and not fixed_moment
        , geo if const_geo else curr_est[N-NUM_COMPS:N]))
        state_ests = np.array([state])
        Ps = np.array([P])
        ps = np.array([np.diag(P)[:N]])
        ws = np.empty((iter, N)) 
        norms = []
        costs = []
        traces = []
        return curr_est, changed_params, N, F, G, P, Q, moving, iter, iters_per_iter if moving else None, state_ests, Ps, ps, ws, norms, costs, traces
    
    # Normalize the moments
    def normalize_moments(self, magnet_params, fixed_pos=False, normalize=False):
        for i in range(self.num_magnets):
            if self.is_euclid:
                magnet_params[i*NUM_COMPS*(2-fixed_pos)+NUM_COMPS*(1-fixed_pos):(i+1)*(self.num_params-NUM_COMPS*fixed_pos)] /= la.norm(magnet_params[i*(self.num_params-NUM_COMPS*fixed_pos)+NUM_COMPS*(1-fixed_pos):(i+1)*(self.num_params-NUM_COMPS*fixed_pos)])
            else:   # theta ranges from 0 to 180 degrees, phi ranges from -180 to 180 degrees
                theta, phi = magnet_params[i*self.num_params+NUM_COMPS*(1-2*fixed_pos):(i+1)*(self.num_params-NUM_COMPS*fixed_pos)]
                flip = False
                if normalize or theta > 180 + EPS or theta < -EPS:
                    theta %= 360
                    flip = theta > 180
                    theta = np.abs((theta % 180) - 180*(theta >= 180))
                if normalize or phi > 180 + EPS or phi < -180 - EPS or flip:
                    phi = ((phi % 360) - 180*flip) % 360
                    phi -= 360*(phi > 180)
                magnet_params[i*self.num_params+NUM_COMPS*(1-2*fixed_pos):(i+1)*(self.num_params-NUM_COMPS*fixed_pos)] = theta, phi
        return magnet_params.reshape(-1, 1)
    
    # Update the state vector for cost and Jacobian functions
    def update_state(self, state, curr_est, fixed_pos, fixed_moment, const_geo):
        if not const_geo:
            state[-NUM_COMPS:] = curr_est[-NUM_COMPS:]
        if not (fixed_pos and fixed_moment):
            indices = np.add.outer(self.num_params * np.arange(self.num_magnets), np.arange(NUM_COMPS if fixed_pos else 0, NUM_COMPS if fixed_moment else self.num_params)).ravel()
            state[indices] = curr_est[:len(curr_est)//2-NUM_COMPS if not const_geo else len(curr_est)//2]
        return state.reshape(-1, 1)
    
    # Update geomagnetic disturbance with IMU data
    def update_geo(self, curr_geo):
        A_cont = self.omega - (np.eye(NUM_COMPS) / self.geo_tau)
        A_disc = la.expm(A_cont * self.dt)
        return A_disc.dot(curr_geo)

    # Compute the inverse using the Cholesky factorization
    def cholesky_inverse(self, A):
        return la.cho_solve(la.cho_factor(A), np.eye(A.shape[0]))
    
    # Runs the Extended Information Filter
    def run_EIF(self, initial_guess=None, lm_guess=None, iter=5000, all_states=False, init_P=None, fixed_pos=False, fixed_moment=False, fixed_geo=False, with_IMU=False, with_geo=False, with_noise=False, plotting=False, plot_inter=20, show_dist=False, verbose=2, moving=False, start_idx=0):
        with_IMU = with_IMU and self.from_file and 'Omega' in self.data[0]
        with_geo = with_geo or fixed_geo or (self.from_file and 'B_recon' not in self.data[self.iter]) or with_IMU
        const_geo = not with_geo or fixed_geo
        show_dist = show_dist and self.num_magnets % 2 == 0 and self.num_magnets > 0
        if verbose == 2:
            print(f"Running Extended Information Filter with {'data from file, ' if self.from_file else ''}{'noise, ' if with_noise else 'no noise, '}{'' if with_geo else 'no '}geo disturbance,")
            print(f"{self.num_magnets} magnet(s), {self.num_boards} board(s), position tau: {round(self.pos_tau, NUM_COMPS)} s, moment tau: {round(self.mom_tau, NUM_COMPS)} s{f', geo tau: {round(self.geo_tau, NUM_COMPS)} s' if with_geo else ''}, dt: {round(self.dt, NUM_COMPS)} s, position std: {round(self.pos_std, NUM_COMPS) if not hasattr(self.pos_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.pos_std)} mm, moment std: {round(self.mom_std, NUM_COMPS) if not hasattr(self.mom_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.mom_std)}{'' if self.is_euclid else ' degrees'},{f' geo std: {round(self.geo_std, NUM_COMPS) if not hasattr(self.geo_std, '__len__') else tuple(round(x, NUM_COMPS) for x in self.geo_std)} µT,' if with_geo else ''} and noise std: {round(self.noise_std, NUM_COMPS)} µT\n")
        curr_est, changed_params, N, F, G, P, Q, moving, iter, iters_per_iter, state_ests, Ps, ps, _, norms, costs, traces = self.init_filter(initial_guess, lm_guess, iter, init_P, fixed_pos, fixed_moment, fixed_geo, with_geo, verbose == 2, moving, start_idx)
        # Invert R
        R_inv = self.cholesky_inverse(self.R)
        state = deepcopy(state_ests[-1])
        # Run the Extended Information Filter
        try:
            if verbose > 0:
                    pbar = tqdm(total=iter)
                    pbar.clear()
                if plotting:
                    hdisplay_img1 = display(display_id='State Plot Display')
                    hdisplay_img2 = display(display_id='Distance Plot Display')
                
                for i in range(iter):
                    prev_est = curr_est
                    for _ in range(1) if not moving else range(iters_per_iter[i]):
                        # Prediction step
                        curr_est = F.dot(curr_est)
                        P = F.dot(P).dot(F.T) + G.dot(Q).dot(G.T)
                        Y = self.cholesky_inverse(P)
                        y = Y.dot(curr_est)
                        # Adjust state vector for cost and Jacobian functions
                        state = self.update_state(state, curr_est, fixed_pos, fixed_moment, const_geo)
                        # Compute the residual
                        residual = -self.cost_function(state, 'eif', with_geo, with_noise)
                        # Compute the Jacobian
                        H = self.kalman_jacobian_function(state, not const_geo, fixed_pos, fixed_moment)
                        # Compute measurement covariance and measurement vector
                        HTR_inv = H.T.dot(R_inv)
                        S = HTR_inv.dot(H)
                        s = HTR_inv.dot(residual + H.dot(curr_est))
                        # Update the information matrix and information vector
                        Y += S
                        y += s
                        P = self.cholesky_inverse(Y)
                        curr_est = P.dot(y)
                    if moving:
                        self.update_data(self.iter + 1, with_geo)

                    # For plotting/debugging
                    state = self.normalize_moments(self.update_state(state, curr_est, fixed_pos, fixed_moment, const_geo), normalize=True)
                    state_ests = np.vstack((state_ests, state))
                    Ps = np.vstack((Ps, [P]))
                    ps = np.vstack((ps, [np.diag(P)[:N]]))
                    norms.append(la.norm((curr_est - prev_est)[:N]))
                    costs.append(la.norm(residual))
                    traces.append(np.trace(P))
                    if verbose > 0:
                        pbar.update(1)
                        pbar.set_description(f"Norm: {norms[-1]:.3e}, Cost: {costs[-1]:.3e}, Trace: {traces[-1]:.3e}")
                    if plotting and (i+1) % plot_inter == 0:
                        figs = self.plotting(states=state_ests[len(state_ests)-max(plot_inter, 20)+1:], running=True, with_true=True, moving=moving, show_dist=show_dist, start_idx=start_idx+len(state_ests)-max(plot_inter, 20)-1)
                        hdisplay_img1.update(figs[0])
                        if show_dist:
                            time.sleep(0.1)
                            hdisplay_img2.update(figs[1])
        except KeyboardInterrupt:
            print("\nInterrupted\n")
                if not (len(state_ests) == len(Ps) == len(ps) and len(norms) == len(costs) == len(traces)):
                    Ps = np.vstack((Ps, [P]))[:i+2]
                    ps = np.vstack((ps, [np.diag(P)[:N]]))[:i+2]
                    norms.append(la.norm((curr_est - prev_est)[:N]))
                    norms = norms[:i+1]
                    costs.append(la.norm(residual))
                    costs = costs[:i+1]
                    traces.append(np.trace(P))
                    traces = traces[:i+1]
        except Exception as e:
            print(f"\nError: {e}\n")
            print(traceback.format_exc())
            sys.exit(1)
        finally:
            if verbose > 0:
                pbar.close()
            if plotting:
                plt.close()
            if verbose == 2:
                print(f"\nOptimal magnet parameters:\n {np.array2string(state_ests[-1][:N-NUM_COMPS*(not const_geo)].reshape(-1, self.num_params), separator=', ', precision=NUM_COMPS)}")
                if with_geo:
                    print(f"Geo Disturbance (µT): \n{np.array2string(state_ests[-1][-NUM_COMPS:] * (1 if self.scale_down else GEO_SCALE), separator=', ', precision=NUM_COMPS)}")
                print()
            return state_ests if all_states else state_ests[-1], changed_params, Ps if all_states else Ps[-1], ps, norms, costs, traces

###################### SENSOR CLASS ######################
    class Sensor:
        def __init__(self, width, length=None, height=1, spacing=SENSOR_WIDTH, center=(0, 0, 0), angles=(0, 0)):
            length = width if length is None else length
            self.width = width
            self.length = length
            self.height = height
            self.center = center
            self.grid_size = self.width * self.length * self.height
            # Set the sensor spacing
            if not hasattr(spacing, "__len__"):
                x_sp = y_sp = spacing
                z_sp = 0
            elif len(spacing) == 2:
                x_sp = spacing[0]
                y_sp = spacing[1]
                z_sp = 0
            elif len(spacing) == 3:
                x_sp, y_sp, z_sp = spacing
            else:
                raise ValueError("Position standard deviation must be a scalar, pair, or triplet")
            # Set the sensor positions
            self.x_sp = x_sp
            self.y_sp = y_sp
            self.z_sp = z_sp
            self.width_len = self.x_sp * (self.width - 1)
            self.length_len = self.y_sp * (self.length - 1)
            self.height_len = self.z_sp * (self.height - 1)
            self.local_positions = self.grid()
            # Sets the sensor and global positions and min/max values
            self.change_rotation(angles)

        # Return sensor information
        def __str__(self):
            s = ""
            s += f"Center: {self.center}\n"
            return s

        # Print sensor information
        def print_params(self):
            s = ""
            s += f"Number of Sensors: {self.grid_size}\n"
            s += f"Sensor Board Size: {self.width} x {self.length} x {self.height} ({self.width_len} x {self.length_len} x {self.height_len} mm^3)\n"
            s += f"Sensor Spacing: ({round(self.x_sp, NUM_COMPS)}, {round(self.y_sp, NUM_COMPS)}, {round(self.z_sp, NUM_COMPS)}) mm\n"
            s += f"Min/Max Values: x: {self.x_min} to {self.x_max}, y: {self.y_min} to {self.y_max}, z: {self.z_min} to {self.z_max}\n"
            return s

        # Generate a grid of sensor positions, centered at the origin, with a spacing of 2.5 mm
        def grid(self):
            sensor_positions = []
            for y in range(self.length):
                for z in range(self.height):
                    for x in range(self.width):
                        pos = np.round([self.x_sp * (x - (self.width - 1) * 0.5), self.y_sp * (y - (self.length - 1) * 0.5), self.z_sp * (z - (self.height - 1) * 0.5)], NUM_PARAMS)
                        sensor_positions.append(pos)
            return np.array(sensor_positions)

        # Rotate the sensor positions based on theta and phi
        def rotate(self):
            sensor_positions = []
            for pos in self.local_positions:
                sensor_positions.append(np.round(self.rot.apply(pos), NUM_PARAMS))
            return np.array(sensor_positions)
        
        # Change the rotation of the sensor positions
        def change_rotation(self, angles_rot=(0, 0)):
            if type(angles_rot) == Rot:
                self.rot = angles_rot
            else:
                self.rot = Rot.from_euler('yz', angles_rot, degrees=True) if len(angles_rot) == 2 else Rot.from_euler('xyz', angles_rot, degrees=True)
            self.normal = self.rot.apply([0, 0, 1])
            self.rotated_positions = self.rotate()
            self.global_positions = np.array([self.center + pos for pos in self.rotated_positions])
            self.set_min_max()
        
        # Change the center of the sensor positions
        def change_center(self, center=(0, 0, 0)):
            self.center = center
            self.global_positions = np.array([self.center + pos for pos in self.rotated_positions])
            self.set_global_min_max()

        # Stores min and max values for each axis of the sensor positions
        def set_min_max(self):
            self.x_min, self.y_min, self.z_min = np.min(self.local_positions, axis=0)
            self.x_max, self.y_max, self.z_max = np.max(self.local_positions, axis=0)
            self.set_global_min_max()
        
        # Stores min and max values for each axis of the global positions
        def set_global_min_max(self):
            self.global_x_min, self.global_y_min, self.global_z_min = self.rot.apply([self.x_min, self.y_min, self.z_min]) + self.center
            self.global_x_max, self.global_y_max, self.global_z_max = self.rot.apply([self.x_max, self.y_max, self.z_max]) + self.center
        
        # Flip the sensor positions along an axis
        def flip_pos(self, axis):
            self.local_positions[:, axis] *= -1
            self.rotated_positions = self.rotate()
            self.global_positions = np.array([self.center + pos for pos in self.rotated_positions])
            self.set_min_max()
        
        # Read the sensor positions from a data array or file
        def read_local_positions(self, data):
            if type(data) == str:
                data = np.genfromtxt(data, delimiter=',', names=True)
            
            for i, x, y, z in data:
                self.local_positions[int(i)] = np.array([x, y, z])*1e3
            self.rotated_positions = self.rotate()
            self.global_positions = np.array([self.center + pos for pos in self.rotated_positions])
            self.set_min_max()

        # Compute the magnetic field (T)
        def field(self, moment_vector, position_vector):
            r = position_vector
            r_norm = la.norm(r)
            if r_norm == 0:
                return np.full(NUM_COMPS, np.inf)
            r_hat = r / r_norm
            B = (3 * r_hat.dot(moment_vector) * r_hat - moment_vector) / r_norm**3
            return B
