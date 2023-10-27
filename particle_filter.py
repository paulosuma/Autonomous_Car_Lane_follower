import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

import random

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # (Default) The whole map
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)


            ## first quadrant
            # x = 
            # y =

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

        ###############

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle


        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####
        particle_weight_dict = {}
        for p in self.particles:
            particle_weight_dict[p] = self.weight_gaussian_kernel(readings_robot, p.read_sensor())
        
        sum_weight  = sum(particle_weight_dict.values())

        for p in self.particles:
            p.weight = (particle_weight_dict[p]/sum_weight)

        ###############
        # pass

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        #low-variance-sampling
        particle_array = []
        for p in self.particles:
            particle_array.append(p)

        r = np.random.uniform(0, 1/self.num_particles)
        c = particle_array[0].weight
        i = 0
        for j in range(self.num_particles):
            U = r + (j*(1/self.num_particles))
            while U > c:
                # print("i am stuck")
                i +=1
                if i>=self.num_particles:
                    i = 0 
                c += particle_array[i].weight
            p = particle_array[i]
            particles_new.append(Particle(x = p.x, y = p.y, maze = self.world, sensor_limit = self.sensor_limit))

        ###############

        # #multinomial resampling
        # weight_array = []
        # particle_array = []
        # for p in self.particles:
        #     weight_array.append(p.weight)
        #     particle_array.append(p)
        # cumm_weights = np.cumsum(weight_array)
        # if cumm_weights[-1] > 0:
        #     cumm_weights = cumm_weights/cumm_weights[-1]

        # for j in range(self.num_particles):
        #     sample = np.random.random()
        #     idx = np.searchsorted(cumm_weights, sample)
        #     p = particle_array[idx]
        #     particles_new.append(Particle(x = p.x, y = p.y, maze = self.world, sensor_limit = self.sensor_limit))

        self.particles = particles_new


    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####
        t_step = 0.01
        t0 = 0.0

        for control in self.control:
            [vr, delta] = control
            for p in self.particles:
                vars = [p.x, p.y, p.heading]
                integrator = ode(vehicle_dynamics).set_integrator('dopri5')
                integrator.set_initial_value(vars, t0).set_f_params(vr, delta)
                [p.x, p.y, p.heading] = integrator.integrate(t_step)
        
        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            self.particleMotionModel()
            reading = self.bob.read_sensor()
            self.updateWeight(reading)
            self.resampleParticle()

            self.world.show_estimated_location(self.particles)
            self.world.show_particles(self.particles)
            self.world.show_robot(self.bob)
            self.world.clear_objects()

            ###############
