# Dummy Introduction to PINN

Welcome to the `dummy_intro` section of our PINNacolada Cookbook! 
This folder contains two basic but essential scripts to get you started with Physics-Informed Neural Networks using PyTorch. 
These scripts are designed to be a straightforward demonstration of PINN concept.

`x_squared.py`: A super simple script demonstrating the PINN approach. It uses a neural network to learn the function 0.5 * x^2 by leveraging its derivative. 
Can we learn the actual function by knowing only its derivative?

`x_squared_with_boundary_condition.py`: An extension of the first script, introducing the concept of boundary conditions. 
The script will generate a bunch of png files with plots of the model's predictions. 
Run the script twice: once with `USE_BOUNDARY_CONDITION = True` and once with `USE_BOUNDARY_CONDITION = False` and compare the plots.
It should become obvious why we always need at least one boundary condition.
