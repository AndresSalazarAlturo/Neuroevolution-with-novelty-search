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
from multiprocessing import Pool
from CTRNN import CTRNN

## World's objects positions
## 80, 50
## 100, 140
## Desire final position for cylinder = 170, 175; 180, 180

## Easier map set up
desired_cylinder_pos = [180, 180]
initial_cylinder_pos = [100, 80]
#~ initial_cylinder_pos = [30, 30]
rectangle_big_horizontal_pos = [180, 130]
rectangle_vertical_pos = [30, 130]

## Hard map set up
#~ desired_cylinder_pos = [180, 180]
#~ initial_cylinder_pos = [140, 40]
#~ rectangle_big_horizontal_pos = [130, 135]
#~ rectangle_vertical_pos = [70, 90]
#~ rectangle_small_horizontal_pos = [100, 55]

## Number of robots
num_robots = 2

## GA parameters
num_gens = 50
## 200 in Gomes - pop size
POPULATION_SIZE = 70
GENOTYPE_SIZE = 712
## Weights, bias bounds
#~ weights_bias_range = np.arange(-1, 1, 0.01)
weights_bias_range = np.arange(0.01, 0.99, 0.01)

## CTRNN related
net_size = 12
step_size = 0.001	## Try to reduce the step size
## This are the number of values from the camera and IR sensors
sensor_inputs = 68

## Archive size
archive_size = 50
dist_metric = 'euclidean_levenshtein'

## Multiprocessing - Processors used
#~ pool = Pool(3)

## Path to save robot behaviour
folder_path = './results/2_epuck_CTRNN_robots_behaviour_5'
## _150Gens_80PopSize_40Archive_Euclidean_DistanceBetweenRobots_and_CylinderTrajectory

class MyEPuck(pyenki.EPuck):
	
	# init EPuck. You can add any args and other code you need
	def __init__(self, genome, net_size, step_size):
		super(MyEPuck, self).__init__()
		## Convert list to array
		genome = np.array(genome)
		
		#~ print(f"Genome: {genome}")

		self.setLedRing(True)

		## Save robot info
		self.xs = []
		self.ys = []

		## CTRNN parameters
		# set up CTRNN
		self.network = CTRNN(size=net_size, step_size=step_size)
		# CTRNN parameters
		self.network.taus = 1 + (2 * genome[0:net_size])
		#~ print(f"Taus values: {self.network.taus}")
		self.network.biases = 9 * (genome[net_size:2*net_size] - 0.5)	
		#~ print(f"Biases values: {self.network.biases}")
		self.network.weights = 9 * (np.reshape(genome[net_size*2:net_size*2+net_size**2], (net_size, net_size)) - 0.5)
		#~ print(f"Network weights values: {self.network.weights}")
		
		## Extract the weights from genome
		genome_input_weights = 10 * (genome[net_size*2+net_size**2:] - 0.5)
		#~ print(f"Input weights values: {genome_input_weights}")
		#~ print(f"Input weights length: {len(genome_input_weights)}")

		## I subtract 4 to net_size as 4 of the 12 neurons will be set to zero
		## Thus, is not necessary to have a big genome. (I think this is correct, ask Chris!!!)
		#~ self.input_weights = genome_input_weights.reshape(net_size - 4, sensor_inputs)
		self.input_weights = genome_input_weights.reshape(net_size - 4, sensor_inputs)
		#~ print(f"Input weights values: {self.input_weights}")
		#~ print(f"Input weights values: {len(self.input_weights)}")

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

		## Loop for euler function
		#~ for i in range(10):
			
			#~ camera_inputs = []
			#~ ir_sensors = []
			#~ ## Extract gray values
			#~ for pixel in range(len(camera_obj)):
				#~ print(f"Camera ToGray pixel #{pixel + 1}: {camera_obj[pixel].toGray()}")
				#~ camera_inputs.append(camera_obj[pixel].toGray())

			#~ print(f"Camera values: {camera_inputs}")

			#~ ## Get robot's raw proximity sensor values
			#~ ir_sensors = self.proximitySensorValues
			#~ print(f"IR sensor values: {type(ir_sensors)}")
			#~ ## Scale sensor values down by factor of 1000
			#~ ir_sensor_inputs = (0.001 * np.array(ir_sensors)).tolist()
			#~ print(f"IR sensor values factored: {ir_sensor_inputs}")

			#~ ## Concatenate the camera and IR inputs
			#~ final_sensor_inputs = camera_inputs + ir_sensor_inputs
			#~ array_final_sensor_inputs = np.array(final_sensor_inputs)
			#~ print(f"LIST Final inputs size: {final_sensor_inputs}") #68! - Testing with 14 inputs
			#~ print(f"ARRAY Final inputs size: {array_final_sensor_inputs}") #68!

			#~ ## Apply input weights - method 1
			#~ CTRNN_inputs_weights = np.dot(self.input_weights, array_final_sensor_inputs).tolist()
			#~ print(f"Dot product value: {CTRNN_inputs_weights}")

			#~ ## Apply input weights - method 2
			#~ CTRNN_inputs_weights = self.input_weights.dot(array_final_sensor_inputs).tolist()
			#~ print(f"Dot product value: {CTRNN_inputs_weights}")

			#~ ## Pad inputs with zeros: first 2 neurons are motor neurons, last 2 neurons are interneurons, 
			#~ ## which do no connect directly to the outside of the CTRNN
			#~ CTRNN_final_inputs_weights = np.array([0, 0] + CTRNN_inputs_weights + [0, 0]) 
			#~ print(f"CTRNN FINAL Input values: {CTRNN_final_inputs_weights}")
			#~ print(f"CTRNN FINAL Input shape: {CTRNN_final_inputs_weights.shape}")

			#~ ## Step the CTRNN
			#~ self.network.euler_step(CTRNN_final_inputs_weights)
		## End of loop for euler function

		camera_inputs = []
		## Extract gray values
		for pixel in range(len(camera_obj)):
			#~ print(f"Camera ToGray pixel #{pixel + 1}: {camera_obj[pixel].toGray()}")
			camera_inputs.append(camera_obj[pixel].toGray())

		#~ print(f"Camera values: {camera_inputs}")

		## Get robot's raw proximity sensor values
		ir_sensors = self.proximitySensorValues
		#~ print(f"IR sensor values: {type(ir_sensors)}")
		## Scale sensor values down by factor of 1000
		ir_sensor_inputs = (0.001 * np.array(ir_sensors)).tolist()
		#~ print(f"IR sensor values factored: {ir_sensor_inputs}")

		## Concatenate the camera and IR inputs
		final_sensor_inputs = camera_inputs + ir_sensor_inputs
		array_final_sensor_inputs = np.array(final_sensor_inputs)
		#~ print(f"LIST Final inputs size: {final_sensor_inputs}") #68! - Testing with 14 inputs
		#~ print(f"ARRAY Final inputs size: {array_final_sensor_inputs}") #68!

		## Apply input weights - method 1
		#~ CTRNN_inputs_weights = np.dot(self.input_weights, array_final_sensor_inputs).tolist()
		#~ print(f"Dot product value: {CTRNN_inputs_weights}")

		## Apply input weights - method 2
		CTRNN_inputs_weights = self.input_weights.dot(array_final_sensor_inputs).tolist()
		#~ print(f"Dot product value: {CTRNN_inputs_weights}")

		## Pad inputs with zeros: first 2 neurons are motor neurons, last 2 neurons are interneurons, 
		## which do no connect directly to the outside of the CTRNN
		CTRNN_final_inputs_weights = np.array([0, 0] + CTRNN_inputs_weights + [0, 0]) 
		#~ print(f"CTRNN FINAL Input values: {CTRNN_final_inputs_weights}")
		#~ print(f"CTRNN FINAL Input shape: {CTRNN_final_inputs_weights.shape}")

		## Step the CTRNN
		self.network.euler_step(CTRNN_final_inputs_weights)
		
		## Motor commands are taken from neurons 0 and 1
		commands = self.network.outputs[:2].tolist()

		scale = 10 # amplification for motor speed commands. 10 may actually be quite small for this robot
		self.leftSpeed = scale * commands[0]
		self.rightSpeed = scale * commands[1]

		## Save pos
		self.xs.append(self.pos[0])
		self.ys.append(self.pos[1])
		
		## CTRNN data
		#~ self.angles.append(self.angle)
		#~ self.sensors.append(sensors)
		#~ self.inputs.append(inputs)
		#~ self.outputs.append(self.network.outputs.tolist())
		#~ self.states.append(self.network.states.tolist())

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
	#~ r_1 = pyenki.RectangularObject(130, 10, 5, 10000000, pyenki.Color(0, 0, 1, 1)) # l1, l2, height, mass colour
	#~ r_1.pos = (rectangle_big_horizontal_pos[0], rectangle_big_horizontal_pos[1])
	#~ ## 0.785
	#~ r_1.angle = 0
	#~ r_1.collisionElasticity = 0
	#~ w.addObject(r_1)
	
	#~ # create a rectangular object and add to world - Vertical one
	#~ r_2 = pyenki.RectangularObject(10, 80, 5, 10000000, pyenki.Color(0, 1, 0, 1)) # l1, l2, height, mass colour
	#~ r_2.pos = (rectangle_vertical_pos[0], rectangle_vertical_pos[1])
	#~ ## 0.785
	#~ r_2.angle = 0
	#~ r_2.collisionElasticity = 0
	#~ w.addObject(r_2)
	
	#~ # create a rectangular object and add to world - Small horizontal one
	#~ r_3 = pyenki.RectangularObject(10, 50, 5, 10000000, pyenki.Color(1, 1, 0, 1)) # l1, l2, height, mass colour
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
	
	# create a rectangular object and add to world - Big horizontal one
	r_1 = pyenki.RectangularObject(80, 10, 5, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_1.pos = (rectangle_big_horizontal_pos[0], rectangle_big_horizontal_pos[1])
	## 0.785
	r_1.angle = 0
	r_1.collisionElasticity = 0
	w.addObject(r_1)
	
	# create a rectangular object and add to world - Vertical one
	r_2 = pyenki.RectangularObject(10, 70, 5, 10000000, pyenki.Color(0, 0, 0, 1)) # l1, l2, height, mass colour
	r_2.pos = (rectangle_vertical_pos[0], rectangle_vertical_pos[1])
	## 0.785
	r_2.angle = 1.6
	r_2.collisionElasticity = 0
	w.addObject(r_2)
	
	# create a cylindrical object and add to world - 30.000
	c = pyenki.CircularObject(15, 15, 30000, pyenki.Color(1, 1, 1, 1)) # radius, height, mass, colour. Color params are red, green, blue, alpha (transparency)
	c.pos = (initial_cylinder_pos[0], initial_cylinder_pos[1]) # set cylinder's position: x, y
	c.collisionElasticity = 0 # floating point value in [0, 1]; 0 means no bounce, 1 means a lot of bounce in collisions
	w.addObject(c) # add cylinder to the world

	## Store cylinder pos
	c_xs = []
	c_ys = []

	## set up robots
	#~ num_robots = 3
	pucks = [0] * num_robots
	
	##Epucks pos
	epucks_pos = [(40, 10), (180, 10)]

	for n in range(num_robots):
		## Create an instance of e-puck class
		e = MyEPuck(genome, net_size, step_size)
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
	total_dis_between_robots = []

	# simulate
	if view:
		w.runInViewer((100, -60), 100, 0, -0.7, 3)
	else:
		for i in range(1200): ##1200
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
				
				##Evaluate maximum distance
				#~ fitness = euclidean_distance(desired_cylinder_pos[0], desired_cylinder_pos[1], c.pos[0], 
													#~ c.pos[1])
													
				#~ print(f"Minimum distance: {fitness}")
				
				## Calculate the average distance between the bots during the simulation
				## Calculate the distance every 200 cycles
				if i % 50 == 0:
					## Calculate the average distance between the two robots
					
					robot_1_and_robot_2_distance = euclidean_distance(pucks[0].xs[-1], pucks[0].ys[-1], pucks[1].xs[-1], pucks[1].ys[-1])
					## List with the distances between the robots during the simulation
					total_dis_between_robots.append(robot_1_and_robot_2_distance)
					
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
	
	## Run GA for a fixed number of generations
	for gen in range(num_gens):
		#~ population_fitness = []
		population_novelty = []

		for ind in range(POPULATION_SIZE):
			print("----------------------------------------")
			print("Generation %i, Genotype %i "%(gen, ind))

			## Get genotype from population
			genotype = population[ind]

			#~ print(f"Run optimization, genotype sent: {genotype}")

			##Evaluate genotype
			pucks, grid, c_xs, c_ys, total_dis_between_robots = run_once(genotype, print_stuff=True, view=False)

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
			cylinder_bd = grid.set_of_visited_rects(c_xs, c_ys)
			cylinder_bd_sorted = sorted(cylinder_bd)
			str_cylinder_bd = str(cylinder_bd_sorted)
			
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
			min_value = 5.67
			max_value = 271.37
	
			normalized_dist_between_robots = [(x - min_value) / (max_value - min_value) for x in total_dis_between_robots]

			## Calculate the average value
			#~ print(f"Original distance between robots: {total_dis_between_robots}")
			if normalized_dist_between_robots != 0:
				avg_normalized_dist_between_robots = sum(normalized_dist_between_robots) / len(normalized_dist_between_robots)
			else:
				normalized_dist_between_robots = 0

			final_bd = (avg_normalized_dist_between_robots, str_cylinder_bd)
			#~ print(f"Final behaviour: {final_bd}")

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
				
			## Stop if fitness is less than 40
			if fitness <= 40:
				found = True
				break
		
		## Get the most novel and least novel behaviour
		most_novel_genome = archive.get_most_novel()
		least_novel_genome = archive.get_least_novel()
		## Get the average novelty
		avg_novelty_archive = archive.get_avg_novelty()
		
		## Store average novelty over generations
		average_novelty_over_time.append(avg_novelty_archive)
		
		if found:
			break
		
		if(gen < num_gens-1):
			population = population_reproduce_novelty(archive.archive, population, POPULATION_SIZE, GENOTYPE_SIZE)
			#~ print(f"New population: {population}")

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
	#~ print(f"Initial population: {population[0]}")
	
	## Run optimization
	most_novel_genome, least_novel_genome, average_novelty_over_time, novelty_archive = run_optimization(population)
	
	print("---------------------------------------------")
	print(f"Most novel genome All Time: {most_novel_genome}")
	print("---------------------------------------------")
	print(f"Least novel genome All Time: {least_novel_genome}")
	print("---------------------------------------------")
	print(f"Average novelty All Time: ", sum(average_novelty_over_time)/len(average_novelty_over_time))
	print("---------------------------------------------")	
	
	## Save the plot for the behaviours in the archive
	plot_behaviours(novelty_archive.archive)
	
	## Save the final archive in .txt file
	save_novelty_archive(novelty_archive.archive)

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
	
#~ folder_name = './results/2_epuck_CTRNN_robots_behaviour_2'
#~ file_name = 'final_novelty_archive.json'

#~ json_data = load_json(folder_name, file_name)
#~ desired_behaviour = 3
#~ for candidate in json_data:
	#~ if candidate['genome_id'] == desired_behaviour:
		#~ tested_genome = candidate['genome']
		#~ break
#~ print(f"{tested_genome}")
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(tested_genome, print_stuff=True, view=True)

#~ params_test = [0] * 280
#~ params_test = [0.5] * 712
#~ e, grid, c_xs, c_ys, total_dis_between_robots = run_once(params_test, print_stuff=True, view=True)
#~ print(f"Max distance between robots: {total_dis_between_robots}")
