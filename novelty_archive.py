import numpy as np
from Levenshtein import distance as leven_distance

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

    def compute_novelty(self, data):
        """
            Data is the new behaviour not the genome. Is just a list.
            :param data: Novelty archive list
        """
        diffs = []
        novelty = 0
        ## Transform data to list
        #~ data = list(data)
        data = str(data)
        if len(self.archive) > 0:
            for e in self.archive:
                ## Transform set to list to use levenshtein distance
                #~ behavior = list(e['data'])
                behavior = str(e['data'])
                #~ print(f"Data: {data}")
                #~ print(f"Behaviour: {behavior}")
                diffs.append(leven_distance(data, behavior))
            novelty = sum(diffs)

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
