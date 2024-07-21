import pyenki
import random
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotille
from Grid import Grid
import math
from new_GA_NS import *
from novelty_archive import *
from multi_layer_nn_controller import *

## World's objects positions
## 80, 50
## 100, 140
## Desire final position for cylinder = 170, 175; 180, 180
initial_cylinder_pos = [140, 40]
rectangle_big_horizontal_pos = [130, 135]
rectangle_vertical_pos = [70, 90]
rectangle_small_horizontal_pos = [100, 55]

## Number of robots
num_robots = 2

## GA parameters
num_gens = 150
POPULATION_SIZE = 50
GENOTYPE_SIZE = 215
## Weights, bias bounds
weights_bias_range = np.arange(-5, 5, 0.5)

## Controller related
num_input_neurons = 3
output_size = 2
inputs_size = 68

## Archive size
archive_size = 40

## Path to save robot behaviour
folder_path = './2_epuck_robots_behaviour_13'
## _150Gens_50PopSize_40Archive_Levenshtein_DistanceBetweenRobots_and_CylinderFinalPos

class MyEPuck(pyenki.EPuck):
	
	# init EPuck. You can add any args and other code you need
	def __init__(self, params, num_input_neurons, output_size, inputs_size):
		super(MyEPuck, self).__init__()
		self.params = params
		self.num_input_neurons = num_input_neurons
		self.output_size = output_size
		self.inputs_size = inputs_size

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

		camera_inputs = []
		## Extract gray values
		for pixel in range(len(camera_obj)):
			#~ print(f"Camera ToGray pixel #{pixel + 1}: {camera_obj[pixel].toGray()}")
			camera_inputs.append(camera_obj[pixel].toGray())

		## Get robot's raw proximity sensor values
		ir_sensors = self.proximitySensorValues
		## Scale sensor values down by factor of 1000
		ir_sensor_inputs = (0.001 * np.array(ir_sensors)).tolist()
		
		## Concatenate the camera and IR inputs
		final_inputs = camera_inputs + ir_sensor_inputs		
		#~ print(f"Final inputs size: {len(final_inputs)}") 68!
		
		## Calculate the motor speeds with multi_layer feed forward controller
		controller_obj = ForwardNeuralNetwork(self.params, self.num_input_neurons, self.output_size, self.inputs_size)
		## Outputs
		commands = controller_obj.forward(final_inputs)
		
		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0][0]
		self.rightSpeed = scale * commands[0][1]

		## Save pos
		self.xs.append(self.pos[0])
		self.ys.append(self.pos[1])

def euclidean_distance(x1, y1, x2, y2):
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

	# create a rectangular object and add to world - Big horizontal one
	r_1 = pyenki.RectangularObject(130, 10, 5, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_1.pos = (rectangle_big_horizontal_pos[0], rectangle_big_horizontal_pos[1])
	## 0.785
	r_1.angle = 0
	r_1.collisionElasticity = 0
	w.addObject(r_1)
	
	# create a rectangular object and add to world - Vertical one
	r_2 = pyenki.RectangularObject(10, 80, 5, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_2.pos = (rectangle_vertical_pos[0], rectangle_vertical_pos[1])
	## 0.785
	r_2.angle = 0
	r_2.collisionElasticity = 0
	w.addObject(r_2)
	
	# create a rectangular object and add to world - Small horizontal one
	r_3 = pyenki.RectangularObject(10, 50, 5, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_3.pos = (rectangle_small_horizontal_pos[0], rectangle_small_horizontal_pos[1])
	## 0.785
	r_3.angle = 1.6
	r_3.collisionElasticity = 0
	w.addObject(r_3)

	# create a cylindrical object and add to world
	c = pyenki.CircularObject(15, 15, 1000, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)	
	#~ c = pyenki.CircularObject(20, 15, 1000, pyenki.Color(0, 0, 0, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
	c.pos = (initial_cylinder_pos[0], initial_cylinder_pos[1]) # set cylinder's position: x, y
	c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	w.addObject(c) # add cylinder to the world

	## Store cylinder pos
	c_xs = []
	c_ys = []

	## set up robots
	#~ num_robots = 2
	pucks = [0] * num_robots
	
	for n in range(num_robots):
		## Create an instance of e-puck class
		e = MyEPuck(genome, num_input_neurons, output_size, inputs_size)
		pucks[n] = e
		e.pos = (n * 50, n * 60)
		#~ e.pos = (n * 0, n * 1)
		#~ e.pos = (n * 130, n * 90)
		e.collisionElasticity = 0
		w.addObject(e)

	#~ print(f"Pucks after being created: {pucks}")

	#~ e = MyEPuck(genome)
	#~ e.pos = (50, 60)
	#~ e.collisionElasticity = 0
	#~ w.addObject(e)

	## Average distance between the robots
	total_dis_between_robots = []
	
	# simulate
	if view:
		w.runInViewer((100, -60), 100, 0, -0.7, 3)
	else:
		for i in range(1000): ##1000
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
				
				## Calculate the average distance between the bots during the simulation
				## Calculate the distance every 200 cycles
				if i % 50 == 0:
					dis_between_robots = euclidean_distance(pucks[0].xs[-1], pucks[0].ys[-1], pucks[1].xs[-1], pucks[1].ys[-1])
					## List with the distances between the robots during the simulation
					total_dis_between_robots.append(dis_between_robots)
					
				#~ dis_between_robots = euclidean_distance(pucks[0].xs[-1], pucks[0].ys[-1], pucks[1].xs[-1], pucks[1].ys[-1])
				#~ total_dis_between_robots.append(dis_between_robots)
				
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

		# return robot and grid
		return pucks, grid, c_xs, c_ys, total_dis_between_robots

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
	archive = NoveltySearchArchive(archive_size, leven_distance)
	
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
			pucks, grid, c_xs, c_ys, total_dis_between_robots = run_once(genotype, print_stuff=True, view=False)
			
			## Final cylinder position
			#~ cylinder_final_pos = (c_xs[-1], c_ys[-1])
			#~ print(f"Cylinder final pos: X:{cylinder_final_pos[0]}, Y:{cylinder_final_pos[1]}")
			
			##Evaluate fitness
			#~ fitness = fitness_calculate_distance(initial_cylinder_pos[0], initial_cylinder_pos[1], cylinder_final_pos[0], 
													#~ cylinder_final_pos[1])

			## Add fitness to population fitness
			#~ population_fitness.append(fitness)				
			
			## Get robot behaviour
			#~ robot_bd = grid.set_of_visited_rects(e.xs, e.ys)
			#~ print(f"Robot behaviour description: {robot_bd}")
			
			#~ print(f"Epuck 1: {pucks}")
			
			## Get robots behaviour	
			#~ robot_1_bd = grid.set_of_visited_rects(pucks[0].xs, pucks[0].ys)
			#~ robot_2_bd = grid.set_of_visited_rects(pucks[1].xs, pucks[1].ys)

			## Here just create a third set with the two different behaviours
			#~ str_robot_1_bd = str(robot_1_bd)
			#~ str_robot_2_db = str(robot_2_bd)
			
			#~ final_bd = {str_robot_1_bd, str_robot_2_db}
			
			## Get cylinder behaviour
			#~ cylinder_bd = grid.set_of_visited_rects(c_xs, c_ys)
			#~ str_cylinder_bd = str(cylinder_bd)
			
			#~ final_bd = {str_robot_1_bd, str_robot_2_db, str_cylinder_bd}
			#~ final_bd = cylinder_bd
			## Get cylinder behaviour as final position (x,y)
			cylinder_final_pos = {c_xs[-1], c_ys[-1]}
			## Sort the cylinder information to keep the order when transforming
			## to string
			#~ print(f"CYlinder final POS: {cylinder_final_pos}")
			cylinder_final_pos_sorted = sorted(cylinder_final_pos)
			str_cylinder_final_pos = str(cylinder_final_pos_sorted)

			## Normalize the distance between robots before calculating the average value
			## Values calculated in the simulation
			min_value = 5.67
			max_value = 271.37
	
			normalized_dist_between_robots = [(x - min_value) / (max_value - min_value) for x in total_dis_between_robots]

			## Calculate the average value
			#~ print(f"Original distance between robots: {total_dis_between_robots}")
			if normalized_dist_between_robots != 0:
				avg_normalized_dist_between_robots = sum(normalized_dist_between_robots) / len(normalized_dist_between_robots)
			else:
				normalized_dist_between_robots = 0

			## Sort the normalized robot's distance value to keep the order when
			## transforming to string
			#~ avg_normalized_dist_between_robots_sorted = sorted(avg_normalized_dist_between_robots)
			str_avg_normalized_dist_between_robots = str(avg_normalized_dist_between_robots)

			final_bd = {str_avg_normalized_dist_between_robots, str_cylinder_final_pos}

			## Here add the behaviour to the archive or not.
			## Add the first behaviour to the archive
			if len(archive.archive) == 0:
				archive.insert_entry(genotype, final_bd, 0, genotype_id)
				archive.add_novelty_to_behaviour(0, genotype_id)
				
				## Add novelty to population novelty
				population_novelty.append(0)
				
				## Update genotype ID
				genotype_id += 1

			else:
				## When there is at least one candidate in the archive
				## This behaviour set is the new behaviour that is going
				## To be compared with the behaviours in the archive
				novelty, diffs = archive.compute_novelty(final_bd)
				#~ print(f"Novelty for {robot_bd} is {novelty}")
				archive.insert_entry(genotype, final_bd, novelty, genotype_id)
				archive.add_novelty_to_behaviour(novelty, genotype_id)
				#~ print("---------------------------------------------")
				#~ print(f"Novelty archive: {archive.archive}")

				## Add novelty to population novelty
				population_novelty.append(novelty)
				
				## Update genotype ID
				genotype_id += 1
		
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

def plot_behaviours(archive):
	"""
		Plot archive behaviours and save them.
		:param archive: Final archive list of dictionaries.
	"""
	
	for candidate in archive:
		
		## Run the most novel candidate to plot robot and cylinder trajectory
		pucks, grid, c_xs, c_ys, total_dis_between_robots = run_once(candidate['genome'], print_stuff=True, view=False)
		
		## Plot robot behaviour
		plt.figure()
		for robot_number in range(num_robots):
			if robot_number == 0:
				trajectory_color = 'blue'
				robot_trajectory_number = f"Robot {robot_number}"
			else:
				trajectory_color = 'green'
				robot_trajectory_number = f"Robot {robot_number}"

			## Plot robot trajectory
			plt.plot(pucks[robot_number].xs, pucks[robot_number].ys, color=trajectory_color, linewidth=0.5, label=robot_trajectory_number)
			## Add a square at the start of the trajectory
			plt.plot(pucks[robot_number].xs[0], pucks[robot_number].ys[0], marker='s', markersize=5, color=trajectory_color, label='Start')
			## Add a circle marker at the end of the line
			plt.plot(pucks[robot_number].xs[-1], pucks[robot_number].ys[-1], marker='o', markersize=5, color=trajectory_color, label='End')
			
		## Cylinder trajectory
		plt.plot(c_xs, c_ys, color='red', linewidth=0.5, label='Cylinder trajectory')
		## Add a square at the start of the trajectory
		plt.plot(c_xs[0], c_ys[0], marker='s', markersize=5, color='red', label='Start')
		## Add a circle marker at the end of the line
		plt.plot(c_xs[-1], c_ys[-1], marker='o', markersize=5, color='red', label='End')
		## Plot grid
		grid.plot_grid()
		plt.title(f"Candidate behaviour ID {candidate['genome_id']}")
		plt.legend(loc='upper right', fontsize='small')
		file_name = f"candidate_behaviour_id_{candidate['genome_id']}"
		plt.savefig(f"{folder_path}/{file_name}")
		plt.close()
		#~ plt.show()

def save_novelty_archive(archive):
	"""
		Save novelty archive as text file.
		:param archive: Final archive list of dictionaries.
	"""
	
	## Convert sets to list for json serialization
	for item in archive:
		item['data'] = list(item['data'])
	
	filepath = folder_path + "/final_novelty_archive.json"
	with open(filepath, 'w') as novelty_file:
		json.dump(archive, novelty_file, indent=4)
		#~ for item in archive:
			#~ novelty_archive_file.write(f"Genome ID: {str(item['genome_id'])}, Genome: {', '.join(item['genome'])}, Data: {', '.join(item['data'])}, Novelty: {item['novelty']}\n")

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
	
	
	## Save the plot for the behaviours in the archive
	plot_behaviours(novelty_archive.archive)
	
	## Save the final archive in .txt file
	save_novelty_archive(novelty_archive.archive)

main()

#### Test Candidate ####
## Path to save robot behaviour
#~ folder_json_path = './2_epuck_robots_behaviour_6'
## Load Json data from file
#~ def load_json(folder_name, file_name):
	#~ path = f"{folder_name}/{file_name}"
	#~ with open(path, 'r') as file:
		#~ data = json.load(file)
	#~ return data
	
#~ def access_data(data, key):
	#~ return data.get(key, "Key not found")
	
#~ folder_name = './2_epuck_robots_behaviour_6'
#~ file_name = 'final_novelty_archive.json'

#~ json_data = load_json(folder_name, file_name)
#~ desired_behaviour = 348
#~ for candidate in json_data:
	#~ print(f"{json_data[candidate]}")
	#~ if candidate['genome_id'] == desired_behaviour:
		#~ tested_genome = candidate['genome']
		#~ break
#~ print(f"{tested_genome}")
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(tested_genome, print_stuff=True, view=True)

#~ params_test = [0] * 215
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(params_test, print_stuff=True, view=True)
#~ print(f"Max distance between robots: {total_dis_between_robots}")
