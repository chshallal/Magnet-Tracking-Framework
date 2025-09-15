# Magnet Tracking Framework

A Python framework for tracking magnet dipoles using simulated sensor arrays. This system implements both Levenberg-Marquardt (LM) optimization and Extended Information Filter (INFO) approaches for real-time magnet position and orientation estimation.

## Core Functions

### 1. `evaluate_field(magnet_params)`

**Purpose**: Computes the magnetic field at all sensor locations given magnet parameters.

**Implementation Details**:
```python
# Position difference matrix: sensors × magnets × 3
pos_matrix = sensors_positions[:, None] - magnet_positions[None]

# Magnetic field calculation using dipole equation
B = (3 * np.vecdot(pos_hat, moment_vecs)[..., None] * pos_hat - moment_vecs) / pos_norm**3
```

### 2. `jacobian_function(magnet_params, ...)`

**Purpose**: Computes analytical derivatives of magnetic field with respect to magnet parameters.

**Implementation Details**:
```python
# Position derivatives (example for x-component)
jac[sensor_offset+a, magnet_offset+b] = (
    ((-3 * m_dot_p / pos_norm_5) if a == b else 0)
    - 3 * moment_vecs[..., a] * pos_matrix[..., b] / pos_norm_5 
    - 3 * moment_vecs[..., b] * pos_matrix[..., a] / pos_norm_5 
    + 15 * pos_matrix[..., a] * pos_matrix[..., b] * m_dot_p / pos_norm_7
)
```

### 3. `run_LM(initial_guess=None, ...)`

**Purpose**: Performs Levenberg-Marquardt (LM) optimization for magnet parameter estimation.

**Algorithm**: Non-linear least squares optimization.

**Usage Example**:
```python
# Basic optimization
optimal_params = system.run_LM()

# With geomagnetic compensation
optimal_params = system.run_LM(with_geo=True, verbose=2)

# From specific initial guess
initial = np.array([[10, 5, 25, 0, 0, 1]])  # [x, y, z, mx, my, mz]
optimal_params = system.run_LM(initial_guess=initial)
```

### 4. `run_EIF(initial_guess=None, ...)`

**Purpose**: Implements Extended Information Filter (INFO)

**Filter Steps** (per iteration):
1. **Prediction**: 
   ```python
   x_pred = F @ x_prev
   P_pred = F @ P @ F.T + G @ Q @ G.T
   ```

2. **Information Update**:
   ```python
   Y = inv(P_pred)  # Information matrix
   y = Y @ x_pred   # Information vector
   ```

3. **Measurement Update**:
   ```python
   Y += H.T @ R_inv @ H
   y += H.T @ R_inv @ (z + H @ x_pred)
   ```

4. **State Estimate**:
   ```python
   P = inv(Y)
   x = P @ y
   ```

## System Architecture

### MagnetTrackingSystem Class

**Initialization Parameters**:
- `num_magnets`: Number of magnet dipoles to track
- `num_boards`: Number of sensor boards
- `scalar_m`: Magnetic moment magnitude
- `is_euclid`: Use Euclidean notation
- `pos_tau/mom_tau/geo_tau`: Process noise time constants
- `dt`: Sampling interval

### Sensor Configuration

The system supports flexible sensor array configurations:
```python
# Single 4×4 sensor board
system = MagnetTrackingSystem(width=4, length=4, height=1)

# Multiple boards with custom positioning
system = MagnetTrackingSystem(
    num_boards=2, 
    center=[(0,0,0), (50,0,0)],  # Board positions
    angles=[(0,0), (0,45)]        # Board orientations
)
```

## Usage Examples

### Basic Magnet Tracking
```python

# Initialize system
system = MagnetTrackingSystem(
    num_magnets=2,
    num_boards=1,
    width=3, length=3, height=1
)

# Run LM optimization
lm_result = system.run_LM(with_geo=True, verbose=2)

# Run EIF 
eif_result = system.run_EIF(
    iter=1000,
    moving=False,
    verbose=2
)
```