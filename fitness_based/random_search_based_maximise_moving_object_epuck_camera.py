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

		## Get robot's raw proximity sensor values
		sensors = self.proximitySensorValues
		## Scale sensor values down by factor of 1000
		ir_inputs = (0.001 * np.array(sensors)).tolist()

		## Concatenate the camera and IR inputs
		final_inputs = inputs + ir_inputs

		## Motor commands are taken from nn_controller function
		commands = self.nn_controller(final_inputs, self.params)
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

		# return robot and grid
		return e, grid, c_xs, c_ys

def plot_behaviours(archive, archive_file=False):
	"""
		Plot archive behaviours and save them.
		:param archive: Final archive list of dictionaries.
	"""
	
	if archive_file:
		for candidate in archive:

			## Run the most novel candidate to plot robot and cylinder trajectory
			e, grid, c_xs, c_ys = run_once(candidate['genome'], print_stuff=True, view=False)
			
			## Plot robot behaviour
			plt.figure()
			## Plot robot trajectory
			plt.plot(e.xs, e.ys, color='blue', linewidth=0.5, label='Robot trajectory')
			## Add a square at the start of the trajectory
			plt.plot(e.xs[0], e.ys[0], marker='s', markersize=5, color='blue', label='Start')
			## Add a circle marker at the end of the line
			plt.plot(e.xs[-1], e.ys[-1], marker='o', markersize=5, color='blue', label='End')
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
			#~ plt.show()
	else:		

		## Archive as tuple

		## Run the most novel candidate to plot robot and cylinder trajectory
		e, grid, c_xs, c_ys = run_once(archive[0], print_stuff=True, view=False)
		
		## Plot robot behaviour
		plt.figure()
		## Plot robot trajectory
		plt.plot(e.xs, e.ys, color='blue', linewidth=0.5, label='Robot trajectory')
		## Add a square at the start of the trajectory
		plt.plot(e.xs[0], e.ys[0], marker='s', markersize=5, color='blue', label='Start')
		## Add a circle marker at the end of the line
		plt.plot(e.xs[-1], e.ys[-1], marker='o', markersize=5, color='blue', label='End')
		## Cylinder trajectory
		plt.plot(c_xs, c_ys, color='red', linewidth=0.5, label='Cylinder trajectory')
		## Add a square at the start of the trajectory
		plt.plot(c_xs[0], c_ys[0], marker='s', markersize=5, color='red', label='Start')
		## Add a circle marker at the end of the line
		plt.plot(c_xs[-1], c_ys[-1], marker='o', markersize=5, color='red', label='End')
		## Plot grid
		grid.plot_grid()
		plt.title(f"Candidate behaviour - Fitness {archive[1]}")
		plt.legend(loc='upper right', fontsize='small')
		file_name = f"candidate_behaviour_fitness_9"
		plt.savefig(f"{folder_path}/{file_name}")
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

def run_random_search(best_params, best_fitness):
	
	print("---\n")
	print("Starting search")
	
	final_genotype_and_fitness = ()
	
	for gen in range(num_gens):
		
		print(f"Generation: {gen}")

		## Create a random genotype
		genotype = create_random_single_parameters_set(weights_bias_range, GENOTYPE_SIZE)

		##Evaluate genotype
		e, grid, c_xs, c_ys = run_once(genotype, print_stuff=True, view=False)

		## Final cylinder position
		cylinder_final_pos = (c_xs[-1], c_ys[-1])
		#~ print(f"Cylinder final pos: X:{cylinder_final_pos[0]}, Y:{cylinder_final_pos[1]}")

		#~ ##Evaluate fitness
		fitness = fitness_calculate_distance(initial_cylinder_pos[0], initial_cylinder_pos[1], cylinder_final_pos[0], 
												cylinder_final_pos[1])

		## Check if these parameters are better
		if fitness > best_fitness:
			best_fitness = fitness
			best_params = genotype
			## Add fitness to list of tuples
			final_genotype_and_fitness = (best_params, best_fitness)
			
	return final_genotype_and_fitness

def main():
	
	## Random search variables
	best_params = []
	best_fitness = float('-inf')
	
	## Run random search
	final_genotype_and_fitness = run_random_search(best_params, best_fitness)
	#~ print(f"test: {final_genotype_and_fitness}")
	print(f"Random Best Genotype: {final_genotype_and_fitness[0]}")
	
	#~ e, grid, c_xs, c_ys = run_once(best_genotype, print_stuff=True, view=True)
	
	## Save the plot for the behaviours in the archive
	plot_behaviours(final_genotype_and_fitness)

## 80, 50
initial_cylinder_pos = [100, 140]

params = [0] * 122
num_gens = 200
GENOTYPE_SIZE = 122
## Weights, bias bounds
weights_bias_range = np.arange(-5, 5, 0.5)

## Path to save robot behaviour
folder_path = './robot_behaviour_random_1'

main()

#### Test Candidate ####
#~ candidate = {'genome_id': 6, 'genome': [4.0, 3.5, 3.5, -1.5, -4.5, 1.5, -4.0, -3.0, 3.5, -4.0, 4.5, 3.5, 3.0, -1.5, 4.5, -2.5, -4.0, 3.0, -0.5, 2.0, -0.5, 1.5, -2.5, -4.5, 1.0, -2.5, 3.0, -4.0, -2.5, -3.5, -4.5, 0.0, -3.0, 0.5, 1.5, -3.0, -3.5, 4.5, 0.0, -4.0, -4.5, 1.5, 1.5, -2.5, 4.0, 0.5, 4.0, -3.0, -3.0, 3.5, -0.5, 2.5, -0.5, -3.0, -4.5, -2.5, -1.5, -3.0, 1.0, 4.0, -1.0, -4.0, 0.5, -3.5, 4.5, -3.0, -1.0, -0.5, -2.5, -4.0, 4.0, -1.5, 4.5, -1.5, -2.5, 3.5, -2.0, 1.5, -1.5, -0.5, -3.5, -2.0, -3.0, 3.0, -1.5, 3.5, 2.0, 0.0, -1.0, 0.0, 2.0, 2.0, 4.0, 4.5, 4.0, -3.0, 0.0, -3.0, -4.5, -0.5, 3.0, 2.0, 2.5, -5.0, 4.5, -1.0, -3.0, 4.0, -2.0, -0.5, 0.0, -4.5, -0.5, 0.5, 2.5, -2.5, -1.5, 0.0, 3.0, 1.5, -3.5, 3.5], 'data': {0, 10, 20, 21, 22, 23}, 'novelty': 51}
#~ e, grid, c_xs, c_ys = run_once(candidate['genome'], print_stuff=True, view=True)

#~ ## Plot robot behaviour
#~ plt.figure()
#~ ## Plot robot trajectory
#~ plt.plot(e.xs, e.ys, color='blue', label='Robot trajectory')
#~ ## Add a square at the start of the trajectory
#~ plt.plot(e.xs[0], e.ys[0], marker='s', markersize=5, color='blue', label='Start')
#~ ## Add a circle marker at the end of the line
#~ plt.plot(e.xs[-1], e.ys[-1], marker='o', markersize=5, color='blue', label='End')
#~ ## Cylinder trajectory
#~ plt.plot(c_xs, c_ys, color='red', label='Cylinder trajectory')
#~ ## Add a square at the start of the trajectory
#~ plt.plot(c_xs[0], c_ys[0], marker='s', markersize=5, color='red', label='Start')
#~ ## Add a circle marker at the end of the line
#~ plt.plot(c_xs[-1], c_ys[-1], marker='o', markersize=5, color='red', label='End')
#~ ## Plot grid
#~ grid.plot_grid()
#~ plt.title(f"Most novel candidate behaviour")
#~ plt.legend(loc='upper right', fontsize='small')
#~ file_name = 'Most_Novel_candidate_behaviour'
#~ plt.savefig(f"{folder_path}/{file_name}")
#~ plt.show()
