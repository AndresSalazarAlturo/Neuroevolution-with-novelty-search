import pyenki
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import plotille
from Grid import Grid
import math
from new_GA_NS import *
from novelty_archive import *

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

		## Get robot's camera values
		#~ print('Cam image: ' + str(self.cameraImage))
		#~ print(len(self.cameraImage), self.cameraImage[0])

		camera_obj = self.cameraImage
		#~ for pixel in range(len(camera_obj)):
			#~ print(f"Camera pixel #{pixel + 1}: {camera_obj[pixel]}")
			#~ print("---------------------------")
			
		#~ print("Camera obj: ", dir(camera_obj))
		#~ print("------------------------------------------------------------")
		#~ print("Color instance: ", dir(camera_obj[0]), "\n To Gray: ", camera_obj[0].toGray())
		#~ print("------------------------------------------------------------")
		#~ print("Other instance: ", dir(camera_obj[0].components), "\n Obj --> ", camera_obj[0].components, "\n Class type --> ", type(camera_obj[0].components))
		#~ print("------------------------------------------------------------")
		#~ print("Index: ", dir(camera_obj[0].components.index), "\n Index Value --> ", camera_obj[0].components.index)
		#~ print("------------------------------------------------------------")
		#~ print("Count: ", dir(camera_obj[0].components.count), "\n Count Value --> ", camera_obj[0].components.count)
		#~ print("------------------------------------------------------------")
		#~ help(camera_obj)

		inputs = []
		## Extract gray values
		for pixel in range(len(camera_obj)):
			#~ print(f"Camera ToGray pixel #{pixel + 1}: {camera_obj[pixel].toGray()}")
			inputs.append(camera_obj[pixel].toGray())
			
		#~ print(f"Inputs: {inputs}")

		## Motor commands are taken from nn_controller function
		commands = self.nn_controller(inputs, self.params)
		#~ print(f"Commands: {commands}")
		
		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0]
		self.rightSpeed = scale * commands[1]

		## Test object
		#~ self.leftSpeed = 5
		#~ self.rightSpeed = 5

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
		for i in range(60):
			## Each sensor's contribution to left motor
			left_speed_command += inputs[i] * params[i]
		## Bias for left motor
		left_speed_command += params[120]

		## Right motor speed
		right_speed_command = 0
		for i in range(60):
			## Each sensor's contribution to right motor
			right_speed_command += inputs[i] * params[60 + i]
		## Bias for right motor
		right_speed_command += params[121]	

		# return motor speed commands to robot's controller
		return [left_speed_command, right_speed_command]

#~ def fitness_calculate_distance(x1, y1, x2, y2):
	#~ """
		#~ Euclidean distance between two points.
		#~ :param x1: Initial position in x.
		#~ :param y1: Initial position in y.
		#~ :param x2: Final position in x.
		#~ :param y2: Final position in y.
		#~ :return: Euclidean distance between two points.
	#~ """
	#~ return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
	c = pyenki.CircularObject(20, 15, 1000, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)	
	#~ c = pyenki.CircularObject(20, 15, 1000, pyenki.Color(0, 0, 0, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
	c.pos = (initial_cylinder_pos[0], initial_cylinder_pos[1]) # set cylinder's position: x, y
	c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
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
		for i in range(2000): ##1000
			w.step(0.1, 3)
			#~ c_xs.append(c.pos[0])
			#~ c_ys.append(c.pos[1])
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
		
		#~ print(f"Robot position: {e.pos}")
		
		## Plot robot trajectory
		#~ plt.figure()
		#~ plt.plot(e.xs, e.ys)
		#~ grid.plot_grid()
		#~ plt.title("Robot trajectory")
		#~ plt.show()

		# return robot and grid
		return e, grid, c_xs, c_ys

## 80, 50
initial_cylinder_pos = [100, 140]

params = [0] * 122
num_gens = 2
POPULATION_SIZE = 10
GENOTYPE_SIZE = 122
## Weights, bias bounds
weights_bias_range = np.arange(-5, 5, 0.5)

def run_optimization(population):
	
	print("---\n")
	print("Starting optimization")
	print("Population Size %i, Genome Size %i"%(POPULATION_SIZE, GENOTYPE_SIZE))

	## list to store average fitness
	#~ average_fitness_over_time = []
	
	## List to store average novelty
	average_novelty_over_time = []
	
	## Genotype ID
	genotype_id = 0
	
	## Create novelty search archive instance
	archive = NoveltySearchArchive(4, leven_distance)
	
	## Run GA for a fixed number of generations
	for gen in range(num_gens):
		#~ population_fitness = []
		population_novelty = []

		for ind in range(POPULATION_SIZE):
			print("----------------------------------------")
			print("Generation %i, Genotype %i "%(gen, ind))

			## Get genotype from population
			genotype = population[ind]
			
			print(f"Run optimization, genotype sent: {genotype}")
			
			##Evaluate genotype
			e, grid, c_xs, c_ys = run_once(genotype, print_stuff=True, view=False)
			
			## Final cylinder position
			#~ cylinder_final_pos = (c_xs[-1], c_ys[-1])
			#~ print(f"Cylinder final pos: X:{cylinder_final_pos[0]}, Y:{cylinder_final_pos[1]}")
			
			##Evaluate fitness
			#~ fitness = fitness_calculate_distance(initial_cylinder_pos[0], initial_cylinder_pos[1], cylinder_final_pos[0], 
													#~ cylinder_final_pos[1])
						
			## Add fitness to population fitness
			#~ population_fitness.append(fitness)				
			
			## Get robot behaviour
			robot_bd = grid.set_of_visited_rects(e.xs, e.ys)
			
			## Here add the behaviour to the archive or not.
			## Add the first behaviour to the archive
			if len(archive.archive) == 0:
				archive.insert_entry(genotype, robot_bd, 0, genotype_id)
				archive.add_novelty_to_behaviour(0, genotype_id)
				
				## Add novelty to population novelty
				population_novelty.append(0)
				
				## Update genotype ID
				genotype_id += 1
			
			else:
				## When there is at least one candidate in the archive
				## This behaviour set is the new behaviour that is going
				## To be compared with the behaviours in the archive
				novelty, diffs = archive.compute_novelty(robot_bd)
				#~ print(f"Novelty for {robot_bd} is {novelty}")
				archive.insert_entry(genotype, robot_bd, novelty, genotype_id)
				archive.add_novelty_to_behaviour(novelty, genotype_id)
				#~ print("---------------------------------------------")
				#~ print(f"Novelty archive: {archive.archive}")
				
				## Add novelty to population novelty
				population_novelty.append(novelty)
				
				## Update genotype ID
				genotype_id += 1
			
			#~ print(f"Population fitness: {population_fitness}")
			
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
			
		## Get the most novel and least novel behaviour
		most_novel_genome = archive.get_most_novel()
		least_novel_genome = archive.get_least_novel()
		## Get the average novelty
		avg_novelty_archive = archive.get_avg_novelty()
		
		#~ print("---------------------------------------------")
		#~ print(f"Most novel genome: {most_novel_genome}")
		#~ print("---------------------------------------------")
		#~ print(f"Least novel genome: {least_novel_genome}")
		#~ print("---------------------------------------------")
		#~ print(f"Average novelty in archive: {avg_novelty_archive}")
		#~ print("---------------------------------------------")
		#~ print(f"Population novelty: {population_novelty}")
		#~ print("---------------------------------------------")
			
		#~ best_fitness, best_fitness_val = population_get_fittest(population, population_fitness)
		#~ average_fitness = population_get_average_fitness(population_fitness)
		#~ print(f"Best Fitness params: {best_fitness}")
		#~ print(f"Best Fitness value: {best_fitness_val}")
		#~ print(f"Average Fitness: {average_fitness}")
		
		## Store average fitness over generations
		#~ average_fitness_over_time.append(average_fitness)
		
		## Store average novelty over generations
		average_novelty_over_time.append(avg_novelty_archive)
		
		if(gen < num_gens-1):
			population = population_reproduce_novelty(archive.archive, population, POPULATION_SIZE, GENOTYPE_SIZE)
			print(f"New population: {population}")

	#~ return best_fitness, best_fitness_val, average_fitness_over_time	
	return most_novel_genome, least_novel_genome, average_novelty_over_time, archive

def main():
	
	## Create initial population
	population = create_random_parameters_set(POPULATION_SIZE, GENOTYPE_SIZE, weights_bias_range)
	#~ print(f"Initial population: {population}")
	
	## Run optimization
	#~ fittest_params, fittest_fitness, average_fitness_over_time = run_optimization(population)
	
	## Run optimization
	most_novel_genome, least_novel_genome, average_novelty_over_time, novelty_archive = run_optimization(population)
	
	print("---------------------------------------------")
	print(f"Most novel genome All Time: {most_novel_genome}")
	print("---------------------------------------------")
	print(f"Least novel genome All Time: {least_novel_genome}")
	print("---------------------------------------------")
	print(f"Average novelty All Time: ", sum(average_novelty_over_time)/len(average_novelty_over_time))
	print("---------------------------------------------")
	squares_explored_most_novel = len(most_novel_genome["data"])
	print(f"Explored squared by Most novel candidate: {squares_explored_most_novel}")
	print("---------------------------------------------")
	squares_explored_least_novel = len(least_novel_genome["data"])
	print(f"Explored squared by LEAST novel candidate: {squares_explored_least_novel}")
	print("---------------------------------------------")
	print(f"Final novelty archive: {novelty_archive.archive}")
	
	#~ print("------------------------------------")
	#~ print(f"Best Fitness Params All Time: {fittest_params}")
	#~ print(f"Best Fitness Value All Time: {fittest_fitness}")
	#~ print(f"Average Fitness Per generation All Time: {average_fitness_over_time}")
	#~ print("Average Fitness All Time: ", sum(average_fitness_over_time)/len(average_fitness_over_time))
	
	e, grid, c_xs, c_ys = run_once(most_novel_genome['genome'], print_stuff=True, view=True)

main()

## Test Camera information ##

## This are fixed params to test
#~ e, grid, c_xs, c_ys = run_once(params, print_stuff=True, view=False)

# create rectangular world - note that coordinate origin is corner of arena
#~ w = pyenki.WorldWithTexturedGround(200, 200, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # rectangular arena: width, height, (texture file name?), walls colour

# create circular world - note that coordinate origin is at centre of arena
# w = pyenki.WorldWithTexturedGround(400, "dummyFileName", pyenki.Color(1, 0, 0, 1)) # circular arena: radius, (texture file name?), walls colour
	
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
