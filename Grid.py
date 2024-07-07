import numpy as np
import matplotlib.pyplot as plt
import plotille

class Rect:
	
	# init Rect with edge coords
	def __init__(self, x_left, x_right, y_bottom, y_top):
		self.x_left = x_left
		self.x_right = x_right
		self.y_bottom = y_bottom
		self.y_top = y_top
		
	# check if x, y coordinate lies within rect
	# note that I use >= on one side, and only < on the other - this is so that 
	# a single point will not be detected in 2 adjacent Rects	
	def is_in_rect(self, x, y):
		return x >= self.x_left and x < self.x_right and y >= self.y_bottom and y < self.y_top
	
	# draw rect in active matplotlib figure
	def plot_rect(self):
		plt.plot([self.x_left, self.x_left, self.x_right, self.x_right, self.x_left], 
		         [self.y_bottom, self.y_top, self.y_top, self.y_bottom, self.y_bottom])

	# draw rect in plotille terminal figure 
	def plotille_rect(self, fig):
		fig.plot([self.x_left, self.x_left, self.x_right, self.x_right, self.x_left], 
		         [self.y_bottom, self.y_top, self.y_top, self.y_bottom, self.y_bottom], lc=100)		

class Grid:
	
	# init Grid with edge coords and number of rects in x- and y-axes
	def __init__(self, x_left, x_right, y_bottom, y_top, x_num, y_num):
		self.x_left = x_left
		self.x_right = x_right
		self.y_bottom = y_bottom
		self.y_top = y_top
		self.x_num = x_num
		self.y_num = y_num
		self.rects = []
		self.setup_rects()
		
	# check to see if x, y coordinate lies within grid
	def is_in_grid(self, x, y):
		return x >= self.x_left and x <= self.x_right and y >= self.y_bottom and y <= self.y_top
	
	# set up the grid 
	def setup_rects(self):
		xs = np.linspace(self.x_left, self.x_right, self.x_num+1)
		ys = np.linspace(self.y_bottom, self.y_top, self.y_num+1)
		
		# grid will be built in columns, from left to right, and from bottom upwards
		for i in range(len(xs)-1):
			for j in range(len(ys)-1):
				self.rects.append(Rect(xs[i], xs[i+1], ys[j], ys[j+1]))
		
	# draw grid in active matplotlib figure
	def plot_grid(self):
		for r in self.rects:
			r.plot_rect()

	# draw grid in plotille terminal figure 
	def plotille_grid(self, fig):
		for r in self.rects:
			r.plotille_rect(fig)
				
	# given lists of x and y coordinates (which together specify a trajectory) and return the set of grid rectangles which the trajectory passes through
	def set_of_visited_rects(self, xs, ys):
		inds = set()
		for x, y in zip(xs, ys):
			found = False
			i = 0
			while i < len(self.rects) and not found:
				if self.rects[i].is_in_rect(x, y):
					inds.add(i)
					found = True
				i += 1
		return inds
					

