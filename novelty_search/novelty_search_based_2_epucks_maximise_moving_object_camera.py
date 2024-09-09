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
from multi_layer_Forward_nn_controller import *

## World's objects positions
## Desire final position for cylinder = 170, 175; 180, 180

## Easier map set up
desired_cylinder_pos = [180, 180]
initial_cylinder_pos = [100, 80]
initial_cylinder_pos = [130, 80]
initial_cylinder_pos = [130, 30]
initial_cylinder_pos = [100, 50]
rectangle_big_horizontal_pos = [180, 130]
rectangle_vertical_pos = [30, 130]
small_vertical_pos = [65,170]
vertical_big_wall = [65, 50]
funnel_1 = [160, 40]
funnel_2 = [40, 40]

## Hard map set up
#~ desired_cylinder_pos = [180, 180]
#~ initial_cylinder_pos = [140, 40]
#~ rectangle_big_horizontal_pos = [130, 135]
#~ rectangle_vertical_pos = [70, 90]
#~ rectangle_small_horizontal_pos = [100, 55]

## Number of robots
num_robots = 2

## Controller related
num_nn_neurons = [20, 10, 6, 2]
inputs_size = 68

genotype_size = (inputs_size * num_nn_neurons[0]) + (num_nn_neurons[0] * num_nn_neurons[1]) + (num_nn_neurons[1] * num_nn_neurons[2]) + (num_nn_neurons[2] * num_nn_neurons[3]) +\
                num_nn_neurons[0] + num_nn_neurons[1] + num_nn_neurons[2] + num_nn_neurons[3]

## GA parameters
num_gens = 80
## 200 in Gomes - pop size
POPULATION_SIZE = 100
GENOTYPE_SIZE = genotype_size
## Weights, bias bounds
#~ weights_bias_range = np.arange(-5, 5, 0.5)
weights_bias_range = np.arange(0.01, 0.99, 0.01)

## Archive size
archive_size = 100
dist_metric = 'cylinder_only'

## Path to save robot behaviour
folder_path = './results/2_epuck_Forward_robots_behaviour_34'
## _80Gens_100PopSize_100Archive_CylinderPosXY_HardMap_Bounded

class MyEPuck(pyenki.EPuck):
	
	# init EPuck. You can add any args and other code you need
	def __init__(self, params, num_nn_neurons, inputs_size):
		super(MyEPuck, self).__init__()

		self.num_nn_neurons = num_nn_neurons
		self.inputs_size = inputs_size
		
		self.setLedRing(True)

		## Save robot info
		self.xs = []
		self.ys = []
		
		## Transform the genotype values to a range of -5 to 5
		genome_arr = np.array(params)
		self.genome = 10 * (genome_arr - 0.5)
		#~ print(f"Genotype: {genome}")

	# the EPuck's controller. You can't add args to this method, so any parameters you require must be set in init or other code
	def controlStep(self, dt):

		camera_obj = self.cameraImage

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
		controller_obj = ForwardNeuralNetwork(self.genome, self.num_nn_neurons, self.inputs_size)
		## Outputs
		commands = controller_obj.forward(final_inputs)

		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0][0]
		self.rightSpeed = scale * commands[0][1]

		## Save pos
		self.xs.append(self.pos[0])
		self.ys.append(self.pos[1])
		
def plot_archive_behaviours(archive, default_title = "Candidate behaviour ID", default_file_name = "candidate_behaviour_id",
							pucks=None, grid=None, c_xs=None, c_ys=None, genome_id=None):
	"""
		Plot archive behaviours and save them. Also, plots best solution when found.
		:param archive: Final archive list of dictionaries.
		:param default_title: Name of the default plots - Archive plots.
		:param default_file_name: Name of default files - Archive files.
		:param pucks: e-pucks objects sent to plot best solution.
		:param grid: grid sent to plot best solution.
		:param c_xs: Cylinder x coordinates for best solution.
		:param c_ys: Cylinder y coordinates for best solution.
	"""
	
	if default_title == "Candidate behaviour ID":
		for candidate in archive:
			
			## Run the most novel candidate to plot robot and cylinder trajectory
			pucks, grid, c_xs, c_ys, total_dis_between_robots = run_once(candidate['genome'], print_stuff=True, view=False)
			
			## Plot robot behaviour
			plt.figure()
			for robot_number in range(num_robots):
				if robot_number == 0:
					trajectory_color = 'blue'
					robot_trajectory_number = f"Robot {robot_number+1}"
				elif robot_number == 1:
					trajectory_color = 'green'
					robot_trajectory_number = f"Robot {robot_number+1}"
				else:
					trajectory_color = 'purple'
					robot_trajectory_number = f"Robot {robot_number+1}"

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
			plt.title(f"{default_title} {candidate['genome_id']}")
			plt.legend(loc='upper left', fontsize='small')
			file_name = f"{default_file_name}_{candidate['genome_id']}"
			plt.savefig(f"{folder_path}/{file_name}")
			plt.close()
			#~ plt.show()
	
	else:

		## Plot best solution robots behaviour
		plt.figure()
		for robot_number in range(num_robots):
			if robot_number == 0:
				trajectory_color = 'blue'
				robot_trajectory_number = f"Robot {robot_number+1}"
			elif robot_number == 1:
				trajectory_color = 'green'
				robot_trajectory_number = f"Robot {robot_number+1}"
			else:
				trajectory_color = 'purple'
				robot_trajectory_number = f"Robot {robot_number+1}"

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
		plt.title(f"{default_title} {genome_id}")
		plt.legend(loc='upper left', fontsize='small')
		file_name = f"{default_file_name}_{genome_id}"
		plt.savefig(f"{folder_path}/{file_name}")
		plt.close()
		#~ plt.show()

def save_novelty_archive(archive, file_name="final_novelty_archive.json"):
	"""
		Save novelty archive as text file.
		:param archive: Final archive list of dictionaries.
	"""
	
	## Convert sets to list for json serialization
	#~ for item in archive:
		#~ item['data'] = list(item['data'])
	
	filepath = folder_path + f"/{file_name}"
	with open(filepath, 'w') as novelty_file:
		json.dump(archive, novelty_file, indent=4)
		
def plot_fitnes_over_time(fitness_overtime):
	"""
		Plot and save fitness overtime
		:param fitness_over_time: Best fitness per generation
	"""
	plt.plot(fitness_overtime, marker='o')
	plt.title('fitness overtime')
	
	file_name = "Fitness overtime"
	plt.savefig(f"{folder_path}/{file_name}")
	plt.close()

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

	##############################
	######### Hard world #########
	##############################

	#~ # create a rectangular object and add to world - Big horizontal one
	#~ r_1 = pyenki.RectangularObject(130, 10, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	#~ r_1.pos = (rectangle_big_horizontal_pos[0], rectangle_big_horizontal_pos[1])
	#~ ## 0.785
	#~ r_1.angle = 0
	#~ r_1.collisionElasticity = 0
	#~ w.addObject(r_1)
	
	#~ # create a rectangular object and add to world - Vertical one
	#~ r_2 = pyenki.RectangularObject(10, 80, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	#~ r_2.pos = (rectangle_vertical_pos[0], rectangle_vertical_pos[1])
	#~ ## 0.785
	#~ r_2.angle = 0
	#~ r_2.collisionElasticity = 0
	#~ w.addObject(r_2)
	
	#~ # create a rectangular object and add to world - Small horizontal one
	#~ r_3 = pyenki.RectangularObject(10, 50, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	#~ r_3.pos = (rectangle_small_horizontal_pos[0], rectangle_small_horizontal_pos[1])
	#~ ## 0.785
	#~ r_3.angle = 1.6
	#~ r_3.collisionElasticity = 0
	#~ w.addObject(r_3)

	#~ # create a cylindrical object and add to world - 30.000
	#~ c = pyenki.CircularObject(15, 15, 30000, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
	#~ c.pos = (initial_cylinder_pos[0], initial_cylinder_pos[1]) # set cylinder's position: x, y
	#~ c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	#~ w.addObject(c) # add cylinder to the world
	
	##############################
	######## Easier world ########
	##############################

	# create a rectangular object and add to world - horizontal 1
	r_1 = pyenki.RectangularObject(80, 10, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_1.pos = (rectangle_big_horizontal_pos[0], rectangle_big_horizontal_pos[1])
	## 0.785
	r_1.angle = 0
	r_1.collisionElasticity = 0
	w.addObject(r_1)

	# create a rectangular object and add to world - horizontal 2
	r_2 = pyenki.RectangularObject(10, 70, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_2.pos = (rectangle_vertical_pos[0], rectangle_vertical_pos[1])
	## 0.785
	r_2.angle = 1.6
	r_2.collisionElasticity = 0
	w.addObject(r_2)
	
	# create a rectangular object and add to world - Vertical small one
	r_s = pyenki.RectangularObject(10, 65, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_s.pos = (small_vertical_pos[0], small_vertical_pos[1])
	## 0.785
	r_s.angle = 0
	r_s.collisionElasticity = 0
	w.addObject(r_s)
	
	#~ # create a rectangular object and add to world - Vertical 2
	#~ r_3 = pyenki.RectangularObject(10, 125, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	#~ r_3.pos = (vertical_big_wall[0], vertical_big_wall[1])
	#~ ## 0.785
	#~ r_3.angle = 0
	#~ r_3.collisionElasticity = 0
	#~ w.addObject(r_3)

	## Create funnel 1
	f_1 = pyenki.RectangularObject(10, 140, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	f_1.pos = (funnel_1[0], funnel_1[1])
	f_1.angle = 0.5
	f_1.collisionElasticity = 0
	w.addObject(f_1)

	## Create funnel 2
	f_2 = pyenki.RectangularObject(10, 130, 15, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	f_2.pos = (funnel_2[0], funnel_2[1])
	f_2.angle = -0.4
	f_2.collisionElasticity = 0
	w.addObject(f_2)

	# create a cylindrical object and add to world - 30.000
	c = pyenki.CircularObject(15, 15, 30000, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
	c.pos = (initial_cylinder_pos[0], initial_cylinder_pos[1]) # set cylinder's position: x, y
	c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	w.addObject(c) # add cylinder to the world

	## Store cylinder pos
	c_xs = []
	c_ys = []

	## set up robots
	#~ num_robots = 2
	pucks = [0] * num_robots
	
	##Epucks pos
	epucks_pos = [(60, 10), (140, 10)]
	##Epucks pos for medium map
	#~ epucks_pos = [(80, 10), (180, 10)]
	##Epucks pos for medium-hard map
	#~ epucks_pos = [(40, 10), (160, 10)]
	
	for n in range(num_robots):
		## Create an instance of e-puck class
		e = MyEPuck(genome, num_nn_neurons, inputs_size)
		pucks[n] = e
		## Test
		e.pos = (epucks_pos[n][0], epucks_pos[n][1])
		## My default position
		#~ e.pos = (n * 50, n * 60)
		e.angle = 1.5
		#~ e.pos = (n * 200, n * 10)
		#~ e.pos = (n + 1 * 50, n + 1 * 60)
		#~ e.pos = (n + 1 * 100, n + 1 * 40)
		#~ e.pos = (n * 0, n * 1)
		#~ e.pos = (n * 130, n * 90)
		e.collisionElasticity = 0
		w.addObject(e)

	## Average distance between the robots
	#~ total_dis_between_robots = []
	total_dis_between_robots = [0]
	
	# simulate
	if view:
		w.runInViewer((100, -60), 100, 0, -0.7, 3)
	else:
		for i in range(1000): ##1200
			w.step(0.1, 3)
			
			if print_stuff:
				
				c_xs.append(c.pos[0])
				c_ys.append(c.pos[1])
				
				## Calculate the average distance between the bots during the simulation
				## Calculate the distance every 200 cycles
				#~ if i % 20 == 0:
					## Calculate the average distance between the two robots
					
					#~ robot_1_and_robot_2_distance = euclidean_distance(pucks[0].xs[-1], pucks[0].ys[-1], pucks[1].xs[-1], pucks[1].ys[-1])
					## List with the distances between the robots during the simulation
					#~ total_dis_between_robots.append(robot_1_and_robot_2_distance)
					
				#~ print(f"Total dist between robots: {total_dis_between_robots}")

		# return robot and grid
		return pucks, grid, c_xs, c_ys, total_dis_between_robots

def run_optimization(population):
	
	print("---\n")
	print("Starting optimization")
	print("Population Size %i, Genome Size %i"%(POPULATION_SIZE, GENOTYPE_SIZE))

	## Good candidate found
	found = False
	
	## List to store average novelty
	average_novelty_over_time = []
	
	## Genotype ID
	genotype_id = 0
	
	## Create novelty search archive instance
	archive = NoveltySearchArchive(archive_size, dist_metric)
	
	## Fitness over time
	fitness_over_time = []
	
	## Run GA for a fixed number of generations
	for gen in range(num_gens):

		population_novelty = []
		
		## Save generation trajectories
		gen_trajectories = []

		for ind in range(POPULATION_SIZE):
			print("----------------------------------------")
			print("Generation %i, Genotype %i "%(gen, ind))

			## Get genotype from population
			genotype = population[ind]

			##Evaluate genotype
			pucks, grid, c_xs, c_ys, total_dis_between_robots = run_once(genotype, print_stuff=True, view=False)
			
			## Get cylinder trajectory
			temp_cylinder_trajectory = (c_xs, c_ys)
			gen_trajectories.append(temp_cylinder_trajectory)
			#~ print(f"Temp trajectory: {temp_cylinder_trajectory}")
			
			## Get cylinder behaviour
			#~ cylinder_bd = grid.set_of_visited_rects(c_xs, c_ys)
			#~ print(f"Cylinder NO sorted trajectory: {cylinder_bd}")
			#~ cylinder_bd_sorted = sorted(cylinder_bd)
			#~ print(f"Cylinder sorted trajectory: {cylinder_bd_sorted}")
			#~ list_cylinder_bd = list(cylinder_bd)
			
			## Final cylinder position
			cylinder_final_pos = (c_xs[-1], c_ys[-1])
			#~ print(f"Cylinder final pos: X:{cylinder_final_pos[0]}, Y:{cylinder_final_pos[1]}")
			
			##Evaluate fitness
			fitness = euclidean_distance(desired_cylinder_pos[0], desired_cylinder_pos[1], cylinder_final_pos[0], 
											cylinder_final_pos[1])

			## Minimum and maximum distance of the cylinder to the desired position
			#~ min_cylinder_value = 0
			#~ max_cylinder_value = 233

			## Normalize the distance between cylinder's final pos and cylinder desired pos
			#~ normalized_dist_cylinder = (fitness - min_cylinder_value) / (max_cylinder_value - min_cylinder_value)
			#~ print(f"Normalized cylinder dist: {normalized_dist_cylinder}")

			## Get cylinder behaviour as final position (x,y)
			#~ cylinder_final_pos = {c_xs[-1], c_ys[-1]}
			## Sort the cylinder information to keep the order when transforming
			## to string
			#~ cylinder_final_pos_sorted = sorted(cylinder_final_pos)
			#~ str_cylinder_final_pos = str(cylinder_final_pos_sorted)

			## Normalize the distance between robots before calculating the average value
			## Values calculated in the simulation
			#~ min_value = 5.67
			#~ max_value = 271.37
	
			#~ normalized_dist_between_robots = [(x - min_value) / (max_value - min_value) for x in total_dis_between_robots]

			## Calculate the average value
			#~ print(f"Original distance between robots: {total_dis_between_robots}")
			#~ if normalized_dist_between_robots != 0:
				#~ avg_normalized_dist_between_robots = sum(normalized_dist_between_robots) / len(normalized_dist_between_robots)
			#~ else:
				#~ normalized_dist_between_robots = 0

			#~ final_bd = (avg_normalized_dist_between_robots, str_cylinder_bd, normalized_dist_cylinder)
			final_bd = cylinder_final_pos
			#~ print(f"Final behaviour: {final_bd}")

			## Here add the behaviour to the archive or not.
			## Add the first behaviour to the archive
			if len(archive.archive) == 0:
				archive.insert_entry(genotype, final_bd, 0, genotype_id)
				archive.add_novelty_to_behaviour(0, genotype_id)
				archive.add_fitness_to_behaviour(fitness, genotype_id)
				
				## Add novelty to population novelty
				population_novelty.append(0)
				
				## Update genotype ID
				genotype_id += 1
				
				## To test best solution plot
				#~ novelty = 0

			else:
				## When there is at least one candidate in the archive
				## This behaviour set is the new behaviour that is going
				## To be compared with the behaviours in the archive
				novelty, diffs = archive.compute_novelty(final_bd)
				#~ print(f"Novelty for {robot_bd} is {novelty}")
				archive.insert_entry(genotype, final_bd, novelty, genotype_id)
				archive.add_novelty_to_behaviour(novelty, genotype_id)
				archive.add_fitness_to_behaviour(fitness, genotype_id)
				#~ print("---------------------------------------------")
				#~ print(f"Novelty archive: {archive.archive}")

				## Add novelty to population novelty
				population_novelty.append(novelty)
				
				## Update genotype ID
				genotype_id += 1
				
			## Stop if fitness is less than 20
			if fitness <= 20:
				found = True
				## Save the best genotype - The one that achieve the goal
				best_genotype = [0]
				best_genotype_dict = {"genome_id":genotype_id, "genome":genotype, "data":final_bd, "novelty":novelty, "fitness":fitness, "num_gens":gen}
				best_genotype[0] = best_genotype_dict
				## Save and plot the best result
				save_novelty_archive(best_genotype, file_name="best_solution.json")
				plot_archive_behaviours(best_genotype, default_title = "Best solution behaviour ID", default_file_name = "best_solution_behaviour_id",
										pucks=pucks, grid=grid, c_xs=c_xs, c_ys=c_ys, genome_id=genotype_id)
				break
		
		## Here plot the cylinder behaviours in the generation
		plot_gen_cylinder_behaviors(gen_trajectories, grid, gen)
		#~ print(f"Cylinder trajectories in generation: {gen_trajectories}")
		
		## Get the average novelty
		avg_novelty_archive = archive.get_avg_novelty()
		
		## Store average novelty over generations
		average_novelty_over_time.append(avg_novelty_archive)
		
		## Get the highest fitness value
		best_fitness_genome = archive.get_best_fitness()
		best_fitness_value = best_fitness_genome['fitness']
		fitness_over_time.append(best_fitness_value)
		
		if found:
			break
		
		#~ ## When the optimal solution has not been found
		#~ best_genotype = None
		
		if(gen < num_gens-1):
			population = population_reproduce_novelty(archive.archive, population, POPULATION_SIZE, GENOTYPE_SIZE)
			#~ print(f"New population: {population}")
			
	## Get the most novel and least novel behaviour
	most_novel_genome = archive.get_most_novel()
	least_novel_genome = archive.get_least_novel()	

	#~ return best_fitness, best_fitness_val, average_fitness_over_time	
	return most_novel_genome, least_novel_genome, average_novelty_over_time, archive, fitness_over_time

def plot_gen_cylinder_behaviors(gen_trajectories, grid, gen):
	
	for trajectory in gen_trajectories:
		x_coords, y_coords = trajectory
		
		plt.plot(x_coords, y_coords)
	
	grid.plot_grid()
	plt.title(f"Generation {gen} trajectories")
	file_name = f"generation_{gen}_trajectories"
	plt.savefig(f"{folder_path}/gen_trajectories/{file_name}")
	plt.close()

def main():
	
	#################################################################
	######### Initial population from previous environments #########
	#################################################################

	#~ ## File paths to previous archive solutions
	#~ file_path_best_solution = './results/2_epuck_Forward_robots_behaviour_20/best_solution.json'
	#~ file_final_novelty_archive = './results/2_epuck_Forward_robots_behaviour_20/final_novelty_archive.json'
	
	#~ ## Load JSON data
	#~ best_solution_list, novelty_archive_list = import_json(file_final_novelty_archive, file_path_best_solution)
	
	#~ ## Create initial population from previous environments
	#~ population = initial_pop_from_archive(novelty_archive_list, POPULATION_SIZE, GENOTYPE_SIZE, best_solution_list)

	#############################################
	######### Random initial population #########
	#############################################
	
	## Create initial population
	population = create_random_parameters_set(POPULATION_SIZE, GENOTYPE_SIZE, weights_bias_range)
	#~ print(f"Initial population: {population}")
	
	## Run optimization
	most_novel_genome, least_novel_genome, average_novelty_over_time, novelty_archive, fitness_over_time = run_optimization(population)
	
	print("---------------------------------------------")
	print(f"Most novel genome All Time: {most_novel_genome}")
	print("---------------------------------------------")
	print(f"Least novel genome All Time: {least_novel_genome}")
	print("---------------------------------------------")
	print(f"Average novelty All Time: ", sum(average_novelty_over_time)/len(average_novelty_over_time))
	
	## Save the plot for the behaviours in the archive
	plot_archive_behaviours(novelty_archive.archive)
	
	## Save the final archive in .json file
	save_novelty_archive(novelty_archive.archive)
	
	## Save fitness overtime plot
	plot_fitnes_over_time(fitness_over_time)

if __name__ == '__main__':
	
	main()

#### Test Candidate ####
## Path to save robot behaviour
#~ folder_json_path = './2_epuck_CTRNN_robots_behaviour_2'
## Load Json data from file
#~ def load_json(folder_name, file_name):
	#~ path = f"{folder_name}/{file_name}"
	#~ with open(path, 'r') as file:
		#~ data = json.load(file)
	#~ return data
	
#~ def access_data(data, key):
	#~ return data.get(key, "Key not found")
	
#~ folder_name = './results/2_epuck_Forward_robots_behaviour_5'
#~ file_name = 'best_solution.json'

#~ json_data = load_json(folder_name, file_name)
#~ desired_behaviour = 3522
#~ for candidate in json_data:
	#~ if candidate['genome_id'] == desired_behaviour:
		#~ tested_genome = candidate['genome']
		#~ break
#~ print(f"{tested_genome}")
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(tested_genome, print_stuff=True, view=True)

#~ params_test = [0] * 280
#~ params_test = [0] * GENOTYPE_SIZE
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(params_test, print_stuff=True, view=True)
#~ print(f"Max distance between robots: {total_dis_between_robots}")
