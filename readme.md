# Machine Learning from Coursera

This my solutions to all the assignments of the coursera course tought by Andrew Ng.

in this repository, I've implemented solutions in both 2 languages octave and python and using jupyter
notebook for better project structure and better documentation.

# Trouble shooting

## octave and python favor different 1D vector.

In the course materials and octave, 1D vector always refers to 1D column vector, but in many python libraries they seems to favor 1D row vector.

## visual studio code's jupyter can not start octave kernel

2 possible solutions

1. use remote connection
2. could be virtual env python path problem, the octave kernel config by default doesn't specify which python to use, make sure you are in the virtual env environment or just specify python version in the config.

## octave notebook output keep showing '[?2004l'

It means shell bracked paste mode is on. it not only generate special code to the screen but also cause performance drop.

copy the jupyter notebook config file to jupyter's config folder to solve this problem.
