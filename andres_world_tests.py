import pyenki
import random
import time
import numpy as np

class MyEPuck(pyenki.EPuck):
	
	# init EPuck. You can add any args and other code you need
	def __init__(self, params):
		super(MyEPuck, self).__init__()
		self.timeout = 5 # set timer period
		self.params = params

	# the EPuck's controller. You can't add args to this method, so any parameters you require must be set in init or other code
	def controlStep(self, dt):
		## Get robot's raw proximity sensor values
		sensors = self.proximitySensorValues
		## Scale sensor values down by factor of 1000, and then apply input weights
		inputs = (np.multiply(sensors, self.params) / 1000).tolist()
		## Motor commands are taken from nn_controller function
		commands = nn_controller(inputs, self.params)

		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0]
		self.rightSpeed = scale * commands[1]

		# print some of the robot's data
		if False: # set to True to print data
			print('Control step')
			print('pos: ' + str(self.pos))
			print('IR dists: ' + str(self.proximitySensorDistances))
			print('IR values: ' + str(self.proximitySensorValues))
			print('Cam image: ' + str(self.cameraImage))
			print(len(self.cameraImage), self.cameraImage[0])
			print(id(self), self.pos)
			print()
		
params = [0] * 18
## Left motor params
params[0] = -1
params[1] = -1
params[2] = -0.5
params[3] = -0.5
params[4] = 0.5
params[5] = 0.5
params[6] = 1
params[7] = 1
## Rigth motor params
params[8] = 0.5
params[9] = 0.5
params[10] = 1
params[11] = 1
params[12] = -1
params[13] = -1
params[14] = -0.5
params[15] = -0.5
## Bias terms
params[16] = 1		## For left motor
params[17] = 1		## For right motor
    
def nn_controller(inputs, params):
	"""
		Neural network with forward propagation. No activation function.
		:param inputs: List with sensor values
		:param params: List with weights and bias values
		:return left_speed_command: Left motor speed
		:return right_speed_command: Right motor speed
	"""

	## Left motor speed
	left_speed_command = 0
	for i in range(8):
		## Each sensor's contribution to left motor
		left_speed_command += inputs[i] * params[i]
	## Bias for left motor
	left_speed_command += params[16]
	
	## Right motor speed
	right_speed_command = 0
	for i in range(8):
		## Each sensor's contribution to right motor
		right_speed_command += inputs[i] * params[i]
	## Bias for right motor
	right_speed_command += params[17]	

	# return motor speed commands to robot's controller
	return [left_speed_command, right_speed_command]
	
def run_once(genome, print_stuff=False, view=False):

	# create rectangular world - note that coordinate origin is corner of arena
	w = pyenki.WorldWithTexturedGround(200, 200, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # rectangular arena: width, height, (texture file name?), walls colour
	
	# create a cylindrical object and add to world
	c = pyenki.CircularObject(20, 30, 100, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)	
	c.pos = (100, 100) # set cylinder's position: x, y
	c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	w.addObject(c) # add cylinder to the world

	# set up robot
	e = MyEPuck(genome)
	e.pos = (100, 60)
	e.collisionElasticity = 0
	w.addObject(e)
	
	# simulate
	if view:
		w.runInViewer((100, -60), 100, 0, -0.7, 3)
	else:
		for i in range(1000):
			w.step(0.1, 3)
			if print_stuff:
				print("A robot:", e.pos)

	# return robot
	return e

run_once(params, print_stuff=False, view=True)

# create rectangular world - note that coordinate origin is corner of arena
#~ w = pyenki.WorldWithTexturedGround(200, 200, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # rectangular arena: width, height, (texture file name?), walls colour

# create circular world - note that coordinate origin is at centre of arena
# w = pyenki.WorldWithTexturedGround(400, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # circular arena: radius, (texture file name?), walls colour

# create EPucks on grid and add to world
#~ pucks = []

## Number of robots
#~ num_robots = 1

## Create and configure each robot
#~ for n in range(num_robots):
	#~ ## Create an instance of e-puck class
	#~ e = MyEPuck()
	#~ pucks.append(e)
	#~ ## Set the position of each robot
	#~ e.pos = (n * 30 + 70, 60)
	#~ e.collisionElasticity = 1
	#~ w.addObject(e)
	
## ir_values_list -> e-puck infrared sensor values
## For e-puck in position [0]
#~ print(pucks[0])
#~ first_puck = pucks[0]
#~ ir_values_list = first_puck.proximitySensorValues
#~ left_speed, right_speed = nn_controller(ir_values_list, params)

## For e-puck in position [1]
#~ ir_values_list = pucks[1].proximitySensorValues()
#~ left_speed, right_speed = nn_controller(ir_values_list, params)
	
# create a cylindrical object and add to world
#~ c = pyenki.CircularObject(20, 30, 100, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
#~ c.pos = (100, 100) # set cylinder's position: x, y
#~ c.pos = (100, 100) # set cylinder's position: x, y
#~ c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
#~ w.addObject(c) # add cylinder to the world

# create a rectangular object and add to world
#~ r = pyenki.RectangularObject(40, 20, 5, 10, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
#~ r.pos = (100, 20)
#~ r.angle = 0.785
#~ r.collisionElasticity = 0
#~ w.addObject(r)

#### there are 3 ways to run the simulation:
## Method 1: Run simulation with viewer. Simulation runs until viewer window is closed
#~ w.runInViewer((100, -60), 100, 0, -0.7, 3)

## Method 2a: Run simulation for a period of time
# w.run(20)

## Method 2b: Run simulation for consecutive periods of time in loop - NOT FULLY TESTED - I USE METHOD 3
'''
for i in range(100):
	w.run(1)
	print("Cylinder:", c.pos)
	print("A robot:", pucks[0].pos)
	print("Sensors:", pucks[0].proximitySensorValues)
	print()
'''

## Method 3: Run simulation using world step method in a loop

#~ for i in range(10):
	#~ w.step(0.1, 3) # interval step (dt), physics oversampling
	#~ print("Cylinder:", c.pos)
	#~ print("A robot:", pucks[0].pos)
	#~ print("Sensors:", pucks[0].proximitySensorValues)
	#~ print()
	
	
