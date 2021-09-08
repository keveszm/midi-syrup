# midi-syrup
 A Customisable Genetic Algorithm/Neural Network Hybrid For Melody Generation Based On User Preference

Please read the included paper for further information, the README will be expanded on later.

# Abstract
As music and its creation are becoming more accessible, musicians look for new ways to create it, and to be inspired. One of the main ways this is done is through the use of artificial intelligence. More specifically, genetic algorithms and neural networks. Previous research in this field have either tried using objective fitness functions for their genetic algorithms, or some arbitrary ways to let users score the generated pieces of music. This paper proposes a novel, genetic algorithm-neural network hybrid. A 2D grid of MIDI notes is generated based on user preference, where individuals of the population could move around and ”collect” notes. A MIDI file is then generated from the best individual, which the user can rate from 1 to 7 (1 being the worst-, and 7 being the best score). This rating, along with the MIDI sequence then gets written to a file, which is used as the dataset to train the neural network. Based on initial experiments, the algorithm was able to improve on the quality of the generated melodies based on user feedback.