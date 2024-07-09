import pyenki
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import plotille
from Grid import Grid
import math
from my_GA import *

class MyEPuck(pyenki.EPuck):
	
	# init EPuck. You can add any args and other code you need
	def __init__(self, params):
		super(MyEPuck, self).__init__()
		self.timeout = 5 # set timer period
		self.params = params

		## Save robot info
		self.xs = []
		self.ys = []

	# the EPuck's controller. You can't add args to this method, so any parameters you require must be set in init or other code
	def controlStep(self, dt):
		## Get robot's raw proximity sensor values
		sensors = self.proximitySensorValues
		## Scale sensor values down by factor of 1000
		inputs = (0.001 * np.array(sensors)).tolist()
		## Motor commands are taken from nn_controller function
		commands = self.nn_controller(inputs, self.params)
		#~ print(f"Commands: {commands}")

		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0]
		self.rightSpeed = scale * commands[1]

		## Save pos
		self.xs.append(self.pos[0])
		self.ys.append(self.pos[1])
		
	def nn_controller(self, inputs, params):
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
			right_speed_command += inputs[i] * params[8 + i]
		## Bias for right motor
		right_speed_command += params[17]	

		# return motor speed commands to robot's controller
		return [left_speed_command, right_speed_command]

def fitness_calculate_distance(x1, y1, x2, y2):
	"""
		Euclidean distance between two points.
		:param x1: Initial position in x.
		:param y1: Initial position in y.
		:param x2: Final position in x.
		:param y2: Final position in y.
		:return: Euclidean distance between two points.
	"""
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def run_once(genome, print_stuff=False, view=False):
	
	## Set up grid
	width = 200
	height = 200
	w_num = 10
	h_num = 10
	grid = Grid(0, width, 0, height, w_num, h_num)

	# create rectangular world - note that coordinate origin is corner of arena
	w = pyenki.WorldWithTexturedGround(200, 200, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # rectangular arena: width, height, (texture file name?), walls colour

	# create a cylindrical object and add to world
	c = pyenki.CircularObject(20, 30, 100, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)	
	c.pos = (80, 50) # set cylinder's position: x, y
	c.collisionElasticity = 1 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	w.addObject(c) # add cylinder to the world
	## Store cylinder pos
	c_xs = []
	c_ys = []

	# set up robot
	e = MyEPuck(genome)
	e.pos = (50, 60)
	e.collisionElasticity = 0
	w.addObject(e)

	# simulate
	if view:
		w.runInViewer((100, -60), 100, 0, -0.7, 3)
	else:
		for i in range(1000): ##1000
			w.step(0.1, 3)
			if print_stuff:
				#~ print("A robot:", e.pos)
				#~ print("-----------------")
				#~ print(f"Cylinder pos: {c.pos}")
				#~ print(f"Cylinder pos type: {type(c.pos)}")
				c_xs.append(c.pos[0])
				c_ys.append(c.pos[1])
				#~ print(f"C_xs: {c_xs}")
				#~ print(f"C_ys: {c_ys}")

		#~ if plots:
			## Plot the trajectory in the terminal
			#~ fig = plotille.Figure()
			#~ fig.width = 70
			#~ fig.height = 30
			#~ fig.set_x_limits(min_=0, max_=100)
			#~ fig.set_y_limits(min_=0, max_=100)
			#~ fig.color_mode = 'byte'
			#~ grid.plotille_grid(fig)
			#~ fig.plot(e.xs, e.ys, lc=25)
			#~ print(fig.show())

		# return robot and grid
		return e, grid, c_xs, c_ys

params = [0] * 18
## Avoid objects
#~ ## Left motor params
#~ params[0] = -4
#~ params[1] = -3
#~ params[2] = -1.5
#~ params[3] = -1.5
#~ params[4] = 1.5
#~ params[5] = 1.5
#~ params[6] = 3
#~ params[7] = 3
#~ ## Rigth motor params
#~ params[8] = 1.5
#~ params[9] = 1.5
#~ params[10] = 3
#~ params[11] = 3
#~ params[12] = -3
#~ params[13] = -4
#~ params[14] = -1.5
#~ params[15] = -1.5
#~ ## Bias terms
#~ params[16] = 2		## For left motor
#~ params[17] = 2		## For right motor

## Aggressor
## Left motor params
#~ params[0] = 2
#~ params[1] = 2
#~ params[2] = 2
#~ params[3] = 2
#~ params[4] = 1
#~ params[5] = 1
#~ params[6] = 1
#~ params[7] = 1
#~ ## Rigth motor params
#~ params[8] = 1
#~ params[9] = 1
#~ params[10] = 1
#~ params[11] = 1
#~ params[12] = 2
#~ params[13] = 2
#~ params[14] = 2
#~ params[15] = 2
#~ ## Bias terms
#~ params[16] = 0.5		## For left motor
#~ params[17] = 0.5		## For right motor
		
num_gens = 2
POPULATION_SIZE = 10
GENOTYPE_SIZE = 18
## Weights, bias bounds
weights_bias_range = np.arange(-5, 5, 0.5)

def run_optimization(population):
	
	print("---\n")
	print("Starting optimization")
	print("Population Size %i, Genome Size %i"%(POPULATION_SIZE, GENOTYPE_SIZE))
	
	## list to store average fitness
	average_fitness_over_time = []
	
	## Run GA for a fixed number of generations
	for gen in range(num_gens):
		population_fitness = []
		for ind in range(POPULATION_SIZE):
			print("----------------------------------------")
			print("Generation %i, Genotype %i "%(gen, ind))
		
			## Get genotype from population
			genotype = population[ind]
			
			print(f"Run optimization, genotype sent: {genotype}")
			
			##Evaluate genotype
			e, grid, c_xs, c_ys = run_once(params, print_stuff=True, view=False)
			
			## Final cylinder position
			cylinder_final_pos = (c_xs[-1], c_ys[-1])
			
			##Evaluate fitness
			fitness = fitness_calculate_distance(50, 60, cylinder_final_pos[0], cylinder_final_pos[1])
			
			## Add fitness to population fitness
			population_fitness.append(fitness)
			
			print(f"Population fitness: {population_fitness}")
			
			#~ print(f"c_xs: {c_xs}")
			#~ print(f"c_ys: {c_ys}")

			#~ ## Plot robot trajectory
			#~ plt.figure()
			#~ plt.plot(e.xs, e.ys)
			#~ grid.plot_grid()
			#~ plt.title("Robot trajectory")
			#~ plt.show()

			#~ ## Plot cylinder trajectory
			#~ ## Calculate the Eucliden distance between initial and final position
			#~ final_distance = fitness_calculate_distance(50, 60, cylinder_final_pos[0], cylinder_final_pos[1])
			#~ print(f"Final distance: {final_distance}")
			#~ plt.figure()
			#~ plt.plot(c_xs, c_ys)
			#~ grid.plot_grid()
			#~ plt.title("Cylinder trajectory")
			#~ plt.show()
			
		best_fitness, best_fitness_val = population_get_fittest(population, population_fitness)
		average_fitness = population_get_average_fitness(population_fitness)
		print(f"Best Fitness params: {best_fitness}")
		print(f"Best Fitness value: {best_fitness_val}")
		print(f"Average Fitness: {average_fitness}")
		
		## Store average fitness over generations
		average_fitness_over_time.append(average_fitness)
		
		if(gen < num_gens-1):
			population = population_reproduce(population, population_fitness, GENOTYPE_SIZE)
			print(f"New population: {population}")

	return best_fitness, best_fitness_val, average_fitness_over_time	

def main():
	
	## Create initial population
	population = create_random_parameters_set(POPULATION_SIZE, GENOTYPE_SIZE, weights_bias_range)
	
	## Run optimization
	fittest_params, fittest_fitness, average_fitness_over_time = run_optimization(population)
	
	print("------------------------------------")
	print(f"Best Fitness Params All Time: {fittest_params}")
	print(f"Best Fitness Value All Time: {fittest_fitness}")
	print(f"Average Fitness Per generation All Time: {average_fitness_over_time}")
	print("Average Fitness All Time: ", sum(average_fitness_over_time)/len(average_fitness_over_time))
	
	e, grid, c_xs, c_ys = run_once(fittest_params, print_stuff=True, view=True)

main()


#~ import datetime
#~ today = datetime.date.today()

#~ # List of all attributes and methods
#~ print(dir(today))

#~ # Get detailed help information
#~ help(today)

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

#~ for i in range(100):
	#~ w.run(1)
	#~ print("Cylinder:", c.pos)
	#~ print("A robot:", pucks[0].pos)
	#~ print("Sensors:", pucks[0].proximitySensorValues)
	#~ print()


## Method 3: Run simulation using world step method in a loop

#~ for i in range(10):
	#~ w.step(0.1, 3) # interval step (dt), physics oversampling
	#~ print("Cylinder:", c.pos)
	#~ print("A robot:", pucks[0].pos)
	#~ print("Sensors:", pucks[0].proximitySensorValues)
	#~ print()
