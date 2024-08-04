import numpy as np
from Levenshtein import distance as leven_distance
from difflib import SequenceMatcher 

class NoveltySearchArchive:
    def __init__(self, archive_size, diff_function):
        self.archive_size = archive_size
        self.diff_function = diff_function
        self.archive = []
        self.novelties = []

    def insert_entry(self, genome, data, novelty, genome_id):

        entry = {"genome_id": genome_id,"genome": genome, "data": data}

        ## If archive is not full
        if len(self.archive) < self.archive_size:
            # Add the entry to the archive
            self.archive.append(entry)
            ## Add the current candidate novelty to the novelties list
            self.novelties.append(novelty)

        ## If the archive is full. Get the smaller novelty value and replace that position in the archive with the new
        ## novel solution
        elif novelty > min(self.novelties):
            novelty_index = self.novelties.index(min(self.novelties))
            # print(f"Novelty index: {novelty_index}")
            ## Replace the least novel candidate with a new novel candidate
            self.archive[novelty_index] = entry
            self.novelties[novelty_index] = novelty

    def get_most_novel(self):
        """
            Get the most novel candidate in the archive so far.
        """
        most_novel_genome = max(self.archive, key=lambda x: x['novelty'])
        return most_novel_genome

    def get_least_novel(self):
        """
            Get the least novel candidate in the archive so far.
        """
        least_novel_genome = min(self.archive, key=lambda x: x['novelty'])
        return least_novel_genome

    def get_avg_novelty(self):
        """
            Get the average novelty in the archive.
        """
        return sum(self.novelties)/len(self.novelties)
        
    def sequence_matcher_distance(self, str1, str2):

		#~ str1 = str(s1)
		#~ str2 = str(s2)
		#~ print(f"str1: {str1}")
		#~ print(f"str2: {str2}")

        matcher = SequenceMatcher(None, str1, str2)

        return 1 - matcher.ratio()
        
    def euclidean_distance(self, val_1, val_2):
        """
            Calculate the Euclidean distance between two float values.
            :param val_1: First float value
            :param val_2: Second float value
            :return distance: Euclidean distance
        """
        
        return abs(val_1 - val_2)

    def compute_novelty(self, data):
        """
            Data is the new behaviour not the genome. Is just a set.
            :param data: Novelty archive list
        """
        diffs = []
        novelty = 0
        ## Transform data to list
        #~ data = list(data)
        ## Sort the data to keep the order
        #~ data_sorted = sorted(data)
        #~ data = str(data_sorted)
        
        ## For Euclidean distance, the data is now a tuple. This is when
        ## using average distance between the robots during the simulation
        ## and the euclidean distance between the cylinder and the desired position
        ## Both value are normalized.

        if len(self.archive) > 0:
            for e in self.archive:
                ## Transform set to list to use levenshtein distance
                ## Sort the behavior to keep the order
                #~ behavior_sorted = sorted(e['data'])
                #~ behavior = str(behavior_sorted)
                
                #~ print(f"Data: {data}")
                #~ print(f"Behaviour: {behavior}")
                
                ## Using Levenshtein distance
                #~ diffs.append(leven_distance(data, behavior))
                
                ## Using sequence matcher
                #~ diffs.append(self.sequence_matcher_distance(data, behavior))

                ## Using Euclidean distance for float values ##
                ## The first value of the tuple is the avg distance between the robots
                ## The second value of the tuple is the Euclidean distance between the cylinder
                ## and the desired final position
                
                #~ print(f"Novelty archive: {e}")
                
                if self.diff_function == 'euclidean_only':
                
                    robots_behavior = e['data'][0]
                    #~ print(f"Robots behavior: {robots_behavior}")
                    cylinder_behavior = e['data'][1]
                    #~ print(f"Cylinder behavior: {cylinder_behavior}")

                    robots_diff = self.euclidean_distance(data[0], robots_behavior)
                    cylinder_diff = self.euclidean_distance(data[1], cylinder_behavior)
                    #~ print(f"Cylinder difference: {cylinder_diff}")

                    avg_diff = (robots_diff + cylinder_diff) / len(e['data'])
                    #~ print(f"Average distance: {avg_diff}")
                    diffs.append(avg_diff)
					
                elif self.diff_function == 'euclidean_levenshtein':
					
                    robots_behavior = e['data'][0]
                    #~ print(f"Robots behavior: {robots_behavior}")
                    cylinder_behavior = e['data'][1]
                    #~ print(f"Cylinder trajectory behavior: {cylinder_behavior}")

                    robots_diff = self.euclidean_distance(data[0], robots_behavior)
                    #~ cylinder_diff = self.sequence_matcher_distance(data[1], cylinder_behavior)
                    
                    cylinder_diff = leven_distance(data[1], cylinder_behavior)
                    
                    #~ print(f"Robots distance diff: {robots_diff}")
                    #~ print(f"Cylinder distance diff: {cylinder_diff}")
					
                    avg_diff = (robots_diff + cylinder_diff) / len(e['data'])
                    #~ print(f"Average distance: {avg_diff}")
                    diffs.append(avg_diff)
                
            novelty = sum(diffs)
            #~ print(f"Final novelty: {novelty}")

        return novelty, diffs

    def add_novelty_to_behaviour(self, novelty, genotype_id):
        """
            Add novelty to behaviour based on genotype ID.
            :param novelty: Novelty value for the genotype.
            :param genotype_id: Genotype ID.
        """
        desired_behaviour = genotype_id

        for all_data in self.archive:
            if all_data['genome_id'] == desired_behaviour:
                # print("The desired behaviour is in position: ", self.archive.index(all_data))
                all_data['novelty'] = novelty
                break
