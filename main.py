import random
from midiutil import MIDIFile
import training

appropriate_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
minimum_notes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

batch_size = 32
directions = ["UP", "UR", "RI", "DR", "DO", "DL", "LE", "UL", "ST"]

# SETTINGS START #

search_space_size = 100
number_of_jumps = 10

search_space_key = "C"

# 0 = Ionian, 1 = Dorian, 2 = Phyrgian, 3 = Lydian, 4 = Mixolydian, 5 = Aeolian, 6 = Locrian
search_space_mode = 0

# MAX 7 OCTAVES
range_of_octaves = 1
starting_octave = 3

diatonic_percent = 55
chromatic_percent = 0
silence_percent = 15
holds_percent = 30

population_size = 300
generations = 500
generations_per_training = 10

tournament_size = 8

crossover_rate = 0.9
crossover_chance = 0.5  # For uniform crossover

mutation_chance = 0.2

elitism = 5

tempo = 120
epochs = 200

ind_size = 32

# SETTINGS END #


# Generate the available diatonic notes
def generate_search_space_notes(mode, possible_notes):
	diatonic_notes = []
	chromatic_notes = []

	# Ionian mode
	note_pattern = [1, 1, 0, 1, 1, 1, 0]

	# Dorian mode
	if mode == 1:
		note_pattern = note_pattern[1:] + note_pattern[:1]
	# Phyrgian mode
	elif mode == 2:
		note_pattern = note_pattern[2:] + note_pattern[:2]
	# Lydian mode
	elif mode == 3:
		note_pattern = note_pattern[3:] + note_pattern[:3]
	# Mixolydian mode
	elif mode == 4:
		note_pattern = note_pattern[4:] + note_pattern[:4]
	# Aeolian mode
	elif mode == 5:
		note_pattern = note_pattern[5:] + note_pattern[:6]
	# Locrian mode
	elif mode == 6:
		note_pattern = note_pattern[6:] + note_pattern[:6]

	starting_number = get_minimum_note_number()

	possible_notes = possible_notes[possible_notes.index(starting_number):] + possible_notes[
																			  :possible_notes.index(starting_number)]
	current_note_index = possible_notes.index(starting_number)

	diatonic_notes.append(possible_notes[current_note_index])

	for n in note_pattern:
		if n == 0:
			current_note_index += 1
			if current_note_index < len(possible_notes):
				diatonic_notes.append(possible_notes[current_note_index])
		elif n == 1:
			current_note_index += 2
			if current_note_index < len(possible_notes):
				diatonic_notes.append(possible_notes[current_note_index])

	for n in possible_notes:
		if n not in diatonic_notes:
			chromatic_notes.append(n)

	return diatonic_notes, chromatic_notes


# Get the base note number
def get_minimum_note_number():
	starting_note_number = -1

	if search_space_key == "C":
		starting_note_number = 24
	elif search_space_key == "C#":
		starting_note_number = 25
	elif search_space_key == "D":
		starting_note_number = 26
	elif search_space_key == "D#":
		starting_note_number = 27
	elif search_space_key == "E":
		starting_note_number = 28
	elif search_space_key == "F":
		starting_note_number = 29
	elif search_space_key == "F#":
		starting_note_number = 30
	elif search_space_key == "G":
		starting_note_number = 31
	elif search_space_key == "G#":
		starting_note_number = 32
	elif search_space_key == "A":
		starting_note_number = 33
	elif search_space_key == "A#":
		starting_note_number = 34
	elif search_space_key == "B":
		starting_note_number = 35

	return starting_note_number


# Generate search space
def generate_search_space():
	# Get possible notes
	diatonic_notes, chromatic_notes = generate_search_space_notes(search_space_mode, minimum_notes)

	generated_search_space = []

	chromatic_percent_upper = diatonic_percent + chromatic_percent

	silence_percent_upper = chromatic_percent_upper + silence_percent

	starting_note_number = get_minimum_note_number()

	starting_note_number += (starting_octave - 1) * 12

	for r in range(search_space_size):
		notes_to_add = []

		for c in range(search_space_size):
			if starting_octave <= starting_octave + (range_of_octaves - 1):
				random_octave = random.randint(starting_octave - 1, starting_octave + (range_of_octaves - 1) - 1) * 12
			else:
				random_octave = 0

			# Percent of note
			random_selection = random.randint(1, 100)

			if random_selection <= diatonic_percent:
				random_diatonic_note = random.randint(0, 6)
				notes_to_add.append(diatonic_notes[random_diatonic_note] + random_octave)
			elif diatonic_percent < random_selection <= chromatic_percent_upper:
				random_chromatic_note = random.randint(0, 4)
				notes_to_add.append(chromatic_notes[random_chromatic_note] + random_octave)
			# Add silence
			elif chromatic_percent_upper < random_selection <= silence_percent_upper:
				notes_to_add.append(1)
			# Add hold
			elif silence_percent_upper < random_selection <= 100:
				notes_to_add.append(0)

		generated_search_space.append(notes_to_add)

	return generated_search_space


# Generate population
def generate_population():
	pop = []
	for i in range(population_size):
		# Initialise individual
		individual = [[-1]]
		# Add default fitness (nothing)
		# Initialise genes
		individual_genes = []
		# Add genes
		for g in range(ind_size - 1):
			random_jump = str(random.randint(1, number_of_jumps))
			random_direction = random.randint(0, 8)

			random_direction = directions[random_direction]

			# if the random direction is to stay, no need for a number to jump
			if random_direction == "ST":
				individual_genes.append(random_direction + str(0))
			else:
				individual_genes.append(random_direction + random_jump)

		individual.append(individual_genes)

		pop.append(individual)

	return pop


search_space = generate_search_space()
population = generate_population()

starting_search_position = [random.randint(0, search_space_size - 1), random.randint(0, search_space_size - 1)]
search_space[starting_search_position[0]][starting_search_position[1]] = get_minimum_note_number() + (
		(starting_octave - 1) * 12)


def interpret_individual(individual):
	processed_individual = []

	search_position = starting_search_position.copy()

	# Add first position note
	processed_individual.append(search_space[search_position[0]][search_position[1]])

	for i in range(len(individual)):
		direc = individual[i][:2]
		jumps = int(individual[i][2:])

		if direc == "UP":
			search_position[0] -= jumps
		elif direc == "UR":
			search_position[0] -= jumps
			search_position[1] += jumps
		elif direc == "RI":
			search_position[1] += jumps
		elif direc == "DR":
			search_position[0] += jumps
			search_position[1] += jumps
		elif direc == "DO":
			search_position[0] += jumps
		elif direc == "DL":
			search_position[0] += jumps
			search_position[1] -= jumps
		elif direc == "LE":
			search_position[1] -= jumps
		elif direc == "UL":
			search_position[0] -= jumps
			search_position[1] -= jumps

		# Limit the search boundaries
		if search_position[0] < 0:
			search_position[0] = 0
		elif search_position[0] > search_space_size - 1:
			search_position[0] = search_space_size - 1

		if search_position[1] < 0:
			search_position[1] = 0
		elif search_position[1] > search_space_size - 1:
			search_position[1] = search_space_size - 1

		processed_individual.append(search_space[search_position[0]][search_position[1]])
	return processed_individual


def write_processed_individual_to_midi(ind):
	track = 0
	channel = 0
	midi_time = 0  # In beats
	duration = 0.5  # In beats
	volume = 100

	MyMIDI = MIDIFile(file_format=1)  # One track, defaults to format 1 (tempo track is created
	# automatically)
	MyMIDI.addTempo(track, midi_time, tempo)
	# MyMIDI.addTimeSignature(track, midi_time, 4, 2, 24)

	for i, pitch in enumerate(ind):
		if i == 0 and pitch == 0:
			midi_time += 0.5

		if pitch != 0:
			duration = 0.5

		if i < len(ind) - 1 and pitch != 0:
			# If silence
			if pitch == 1:
				for n in range(i + 1, len(ind)):
					if ind[n] == 0:
						midi_time += duration
					else:
						midi_time += duration
						break
			# If next "note" is hold
			elif ind[i + 1] == 0:
				# Check the length of the hold
				for n in range(i + 1, len(ind)):
					if ind[n] == 0:
						duration += 0.5
					else:
						break
		else:
			# If silence
			if pitch == 1:
				midi_time += duration

		if pitch != 0 and pitch != 1:
			MyMIDI.addNote(track, channel, pitch, midi_time, duration, volume)
			midi_time += duration

	with open("output.mid", "wb") as output_file:
		MyMIDI.writeFile(output_file)


def write_processed_individual_to_train(ind, rating):
	with open('output.tsv', 'a') as f:
		first = str(rating)
		second = str(ind[1])[1:-1].replace(',', '\t').replace(' ', '')

		f.write(first + '\t' + second + '\n')
	return 0


def selection_operator():
	# Tournament selection
	selected_individuals = []
	for k in range(2):
		tournament_individuals = []
		tournament_individual_indexes = []

		for ind in range(tournament_size):
			random_ind_index = random.randint(0, len(population) - 1)
			if random_ind_index not in tournament_individual_indexes:
				tournament_individual_indexes.append(random_ind_index)
				tournament_individuals.append(population[random_ind_index])

		best_score = -2
		best_index = -2

		for i, ind in enumerate(tournament_individuals):
			if ind[0][0] > best_score:
				best_score = ind[0][0]
				best_index = i

		selected_individuals.append(
			[tournament_individual_indexes.pop(best_index), tournament_individuals.pop(best_index)])

	return selected_individuals


def crossover_operator(parents):
	if random.random() <= crossover_chance:
		# Uniform crossover
		child1 = [[-1], []]
		child2 = [[-1], []]
		for i in range(len(parents[0][1][1])):
			if random.random() <= crossover_rate:
				child1[1].append(parents[0][1][1][i])
				child2[1].append(parents[1][1][1][i])
			else:
				child1[1].append(parents[1][1][1][i])
				child2[1].append(parents[0][1][1][i])
	else:
		child1 = [[-1], parents[0][1][1]]
		child2 = [[-1], parents[1][1][1]]

	return [child1, child2]


def mutation_operator(children):
	for c in range(len(children)):
		for g in range(len(children[c][1])):
			if random.random() <= mutation_chance:
				random_jump = str(random.randint(1, number_of_jumps))
				random_direction = random.randint(0, 8)

				random_direction = directions[random_direction]

				# if the random direction is to stay, no need for a number to jump
				if random_direction == "ST":
					children[c][1][g] = random_direction + str(0)
				else:
					children[c][1][g] = random_direction + random_jump

	return children


def replacement_operator(children):
	# Elitism
	elitist_population = []
	for e in range(int((population_size * elitism) / 100)):
		best_individual_index = 0
		best_individual_fitness = 0
		for i, p in enumerate(population):
			if p[0][0] > best_individual_fitness:
				best_individual_index = i
				best_individual_fitness = p[0]

		elitist_population.append(population.pop(best_individual_index))

	# Tournament replacement
	tournament_individuals = []
	tournament_individual_indexes = []

	for ind in range(tournament_size):
		random_ind_index = random.randint(0, len(population) - 1)
		if random_ind_index not in tournament_individual_indexes:
			tournament_individual_indexes.append(random_ind_index)
			tournament_individuals.append(population[random_ind_index])

	for k in range(2):
		worst_score = 100
		best_index = -2

		for i, ind in enumerate(tournament_individuals):
			if ind[0][0] < worst_score:
				worst_score = ind[0][0]
				best_index = i

		if k == 1:
			population.pop(tournament_individual_indexes[best_index] - 1)
		else:
			population.pop(tournament_individual_indexes[best_index])

		tournament_individual_indexes.pop(best_index)
		tournament_individuals.pop(best_index)

	for p in children:
		population.append(p)

	for el in range(len(elitist_population)):
		population.append(elitist_population.pop(0))


def run_algorithm():
	current_generation = 1
	trained = False
	last_best_ind = []
	do_train = False

	while current_generation <= generations:
		best_fitness_score = 0
		current_best_ind = []

		for i, p in enumerate(population):
			if p[0][0] > best_fitness_score:
				best_fitness_score = p[0]
				current_best_ind = p[1]

		print("Running generation", current_generation, "Best Fitness:", best_fitness_score)

		if current_generation == 1:
			do_train = True
			for t in range(10):
				random_individual_index = random.randint(0, len(population) - 1)
				last_best_ind = population[random_individual_index][1]
				random_individual = interpret_individual(population[random_individual_index][1])

				write_processed_individual_to_midi(random_individual)

				user_rating = -1
				# Ask user for rating
				while user_rating < 1 or user_rating > 7:
					print("Please listen to the generated midi file and rate it (1-7)")
					user_rating = int(input())

				individual_to_train = [population[random_individual_index][0], random_individual]

				write_processed_individual_to_train(individual_to_train, user_rating)

		elif current_generation % generations_per_training == 0 or current_generation == generations:
			if current_generation == generations:
				best_individual_index = 0
				best_individual_fitness = 0
				for i, p in enumerate(population):
					if p[0] > best_individual_fitness:
						best_individual_index = i
						best_individual_fitness = p[0]

				best_individual = interpret_individual(population[best_individual_index][1])

				write_processed_individual_to_midi(best_individual)

				print("Final output written to midi")
			elif last_best_ind != current_best_ind:
				best_individual_index = 0
				best_individual_fitness = 0
				for i, p in enumerate(population):
					if p[0] > best_individual_fitness:
						best_individual_index = i
						best_individual_fitness = p[0]

				last_best_ind = population[best_individual_index][1]

				best_individual = interpret_individual(population[best_individual_index][1])

				write_processed_individual_to_midi(best_individual)

				user_rating = -1
				# Ask user for rating
				while user_rating < 1 or user_rating > 7:
					print("Please listen to the generated midi file and rate it (1-7)")
					print(best_individual_fitness, best_individual)
					user_rating = int(input())

				individual_to_train = [population[best_individual_index][0], best_individual]

				write_processed_individual_to_train(individual_to_train, user_rating)

				do_train = True
			else:
				do_train = False

		if do_train:
			is_fitness_valid = False

			while not is_fitness_valid:
				training.train_model(epochs, trained)

				interpreted_population = []
				for p in population:
					interpreted_population.append(interpret_individual(p[1]))

				fitness_list = training.get_fitness_function(interpreted_population)

				for i, p in enumerate(population):
					p[0] = fitness_list[i]

				if 0 not in fitness_list:
					is_fitness_valid = True

			trained = True
			do_train = False

		replacement_operator(mutation_operator(crossover_operator(selection_operator())))

		interpreted_population = []
		for p in population:
			interpreted_population.append(interpret_individual(p[1]))

		fitness_list = training.get_fitness_function(interpreted_population)

		for i, p in enumerate(population):
			p[0] = fitness_list[i]

		current_generation += 1


run_algorithm()
