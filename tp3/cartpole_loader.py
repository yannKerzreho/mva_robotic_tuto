'''
Cartpole loader for optimal control exercises.
Creates a simple cartpole model with visualization.
'''

import numpy as np
import pinocchio as pin
import hppfcl


class CartPoleLoader(object):
    def __init__(self, cart_mass=1.0, pole_mass=1.0, pole_length=1.0):
        """
        Initialize cartpole model.
        
        Args:
            cart_mass (float): Mass of the cart (M)
            pole_mass (float): Mass of the pole (m)
            pole_length (float): Length of the pole (l)
        """
        self.M = cart_mass
        self.m = pole_mass 
        self.l = pole_length
        self.g = 9.81
        
        # Create the robot model
        self.robot = self._build_model()
        
    def _build_model(self):
        """Build the cartpole model using Pinocchio."""
        # Create empty model
        model = pin.Model()
        
        # Add universe joint (fixed base)
        universe_joint_id = 0
        
        # Add prismatic joint for cart translation (x-axis)
        cart_joint_id = model.addJoint(
            universe_joint_id,
            pin.JointModelPX(),
            pin.SE3.Identity(),
            "cart_joint"
        )
        
        # Cart body (box shape)
        cart_inertia = pin.Inertia(
            self.M,
            np.array([0, 0, 0]),  # center of mass at origin
            np.diag([self.M/12 * (0.2**2 + 0.1**2), self.M/12 * (0.4**2 + 0.1**2), self.M/12 * (0.4**2 + 0.2**2)])
        )
        model.appendBodyToJoint(cart_joint_id, cart_inertia, pin.SE3.Identity())
        
        # Add revolute joint for pole rotation (y-axis)
        pole_joint_placement = pin.SE3(np.eye(3), np.array([0, 0, 0.0]))  # joint at top of cart
        pole_joint_id = model.addJoint(
            cart_joint_id,
            pin.JointModelRY(),
            pole_joint_placement,
            "pole_joint"
        )
        
        # Pole body (cylinder shape)
        pole_com = np.array([0, 0, self.l/2])  # center of mass at half length
        pole_inertia = pin.Inertia(
            self.m,
            pole_com,
            np.diag([self.m/12 * self.l**2, self.m * (3 * 0.05**2 + self.l**2)/12, self.m/12 * self.l**2])
        )
        model.appendBodyToJoint(pole_joint_id, pole_inertia, pin.SE3.Identity())
        
        # Create visual model
        visual_model = pin.GeometryModel()
        
        # Cart visual (box)
        cart_shape = hppfcl.Box(0.4, 0.2, 0.1)  # width, height, depth
        cart_geom = pin.GeometryObject(
            "cart_visual",
            cart_joint_id,
            cart_shape,
            pin.SE3.Identity()
        )
        cart_geom.meshColor = np.array([70/256.0, 130/256.0, 180/256.0, 1.0])  # Blue cart
        visual_model.addGeometryObject(cart_geom)
        
        # Pole visual (cylinder)
        pole_shape = hppfcl.Cylinder(0.05, self.l)  # radius, height
        pole_geom = pin.GeometryObject(
            "pole_visual", 
            pole_joint_id,
            pole_shape,
            pin.SE3(np.eye(3), pole_com)  # translate to center of mass
        )
        pole_geom.meshColor = np.array([153/256.0, 101/256.0, 21/256.0, 1.0])  # Gold pole
        visual_model.addGeometryObject(pole_geom)
        
        # Add coordinate frames visualization
        self._add_coordinate_frames(visual_model, cart_joint_id, pole_joint_id)
        
        # Create robot wrapper
        robot = pin.RobotWrapper(model, visual_model=visual_model)
        
        # Set initial configuration (cart at origin, pole upright)
        robot.q0 = np.array([0.0, 0.0])  # [cart_position, pole_angle]
        
        # Create data
        robot.data = robot.model.createData()
        robot.visual_data = robot.visual_model.createData()
        
        return robot
    
    def _add_coordinate_frames(self, visual_model, cart_joint_id, pole_joint_id):
        """Add coordinate frame visualizations."""
        frame_length = 0.2
        frame_radius = 0.01
        
        # X-axis (red)
        x_cyl = hppfcl.Cylinder(frame_radius, frame_length)
        
        # Cart frame
        cart_x_geom = pin.GeometryObject(
            "cart_frame_x",
            cart_joint_id,
            x_cyl,
            pin.SE3(pin.utils.rotate('y', np.pi/2), np.array([frame_length/2, 0, 0.05]))
        )
        cart_x_geom.meshColor = np.array([1, 0, 0, 1])
        visual_model.addGeometryObject(cart_x_geom)
        
        # Y-axis (green)  
        y_cyl = hppfcl.Cylinder(frame_radius, frame_length)
        cart_y_geom = pin.GeometryObject(
            "cart_frame_y",
            cart_joint_id, 
            y_cyl,
            pin.SE3(pin.utils.rotate('x', -np.pi/2), np.array([0, frame_length/2, 0.05]))
        )
        cart_y_geom.meshColor = np.array([0, 1, 0, 1])
        visual_model.addGeometryObject(cart_y_geom)
        
        # Z-axis (blue)
        z_cyl = hppfcl.Cylinder(frame_radius, frame_length) 
        cart_z_geom = pin.GeometryObject(
            "cart_frame_z",
            cart_joint_id,
            z_cyl,
            pin.SE3(np.eye(3), np.array([0, 0, 0.05 + frame_length/2]))
        )
        cart_z_geom.meshColor = np.array([0, 0, 1, 1])
        visual_model.addGeometryObject(cart_z_geom)
        
        # Pole tip frame
        pole_tip_x_geom = pin.GeometryObject(
            "pole_tip_frame_x",
            pole_joint_id,
            x_cyl,
            pin.SE3(pin.utils.rotate('y', np.pi/2), np.array([frame_length/2, 0, self.l]))
        )
        pole_tip_x_geom.meshColor = np.array([1, 0.5, 0.5, 1])
        visual_model.addGeometryObject(pole_tip_x_geom)
        
        pole_tip_z_geom = pin.GeometryObject(
            "pole_tip_frame_z", 
            pole_joint_id,
            z_cyl,
            pin.SE3(np.eye(3), np.array([0, 0, self.l + frame_length/2]))
        )
        pole_tip_z_geom.meshColor = np.array([0.5, 0.5, 1, 1])
        visual_model.addGeometryObject(pole_tip_z_geom)

def loadCartPole(cart_mass=1.0, pole_mass=1.0, pole_length=1.0):
    """
    Load a cartpole model with the specified parameters.
    
    Args:
        cart_mass (float): Mass of the cart (M), default 1.0
        pole_mass (float): Mass of the pole (m), default 1.0  
        pole_length (float): Length of the pole (l), default 1.0
        
    Returns:
        robot: Pinocchio robot wrapper with cartpole model
    """
    loader = CartPoleLoader(cart_mass, pole_mass, pole_length)
    return loader.robot

def nls_cartpole(x, u, Ts, M=1.0, m=1.0, l=1.0, g=9.81):
    """
    Nonlinear cartpole dynamics using forward Euler integration.
    
    Args:
        x: State vector [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        u: Control input (force on cart)
        Ts: Time step
        M: Cart mass
        m: Pole mass  
        l: Pole length
        g: Gravity
        
    Returns:
        x_next: Next state after time step Ts
    """
    xcdot = x[1]
    thetadot = x[3]
    xc = x[0]
    theta = x[2]
    s = np.sin(theta)
    c = np.cos(theta)
    u = np.clip(u, -30, 30)  # Actuator limits
    
    xcddot = (-s*(m*l)*c*thetadot**2 + s*m*c*g + u)/(-m*c**2 + M + m)
    thetaddot = (-s*(m*l)*c*thetadot**2 + s*g*(M+m) + u*c)/(l*(-m*c**2 + M + m))
    
    xc_next = xc + Ts*xcdot
    xcdot_next = xcdot + Ts*xcddot
    theta_next = theta + Ts*thetadot
    thetadot_next = thetadot + Ts*thetaddot
    
    return np.hstack([xc_next, xcdot_next, theta_next, thetadot_next])


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.meshcat_viewer_wrapper import MeshcatVisualizer
    import time

    robot = loadCartPole()
    viz = MeshcatVisualizer(robot, url='classical')

    print("Cartpole model loaded successfully!")
    print(f"Number of joints: {robot.model.nq}")
    print(f"Joint names: {[robot.model.names[i] for i in range(1, robot.model.njoints)]}")
    print(f"Initial configuration: {robot.q0}")
    
    # Display initial configuration
    viz.display(robot.q0)
    
    # Simple animation showing pole swinging
    print("\nAnimating cartpole...")
    for i in range(100):
        t = i * 0.1
        q = np.array([0.5 * np.sin(0.5*t), 0.3 * np.sin(t)])  # cart oscillation, pole swing
        viz.display(q)
        time.sleep(0.05)
        
    print("Animation complete!")
