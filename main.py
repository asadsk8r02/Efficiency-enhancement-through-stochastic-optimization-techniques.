import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from tkinter import *
import os

class BalancingProblemSolver:

    def __init__(self, file):
        self.file = file

    """
    Task One: Read the user file in a problem instance
    """

    def load_file(self):
        return pd.read_csv(self.file, header=None)

    """
    Task Two : Generate an initial Solution
    """

    def initialSolution(self):
        global array, k

        # load the file
        file = self.load_file()

        # Value of K is the first item of the list
        k = file[0][0]

        # Create a list of k partitions
        partitions = [i for i in range(1, (k + 1))]

        # Create Array
        array = file[0][1:].to_numpy()

        initial_solution = []
        for i in range(0, len(array)):
            initial_solution.append(random.choice(partitions))
        return initial_solution

    """
    Task 3: Implement a cost function that calculates the solution.
            :returns cost function value
    """

    @staticmethod
    def calculateCostFunction(solution):

        # Load Array
        weight_sums = np.bincount(solution, weights=array)[1:]  # returns [3,9,9] as a float64 array

        # Calculate the mean
        mean = (sum(weight_sums) / k)

        cost_function = 0
        for weight in weight_sums:
            cost_function += math.sqrt(abs(weight - mean))

        return cost_function

    """
    Task 4: Implement a Feasibility method that returns 1 if the solution is feasible, and 0 otherwise.
    Calculate Feasibility 
    :returns True if each group has atleast k-1 else False
    """

    @staticmethod
    def checkFeasibility(solution_array):
        # Get unique items and their counts
        uni, count = np.unique(solution_array, return_counts=True)

        if count.min() < (k - 1):
            return 1
        else:
            return 0

    """
    Write to csv: This function writes the generated result to a csv file in a format specified in the problem.
    """

    def write_to_csv(self, solution_array):
        # Calculate cost function
        temp_array = solution_array.tolist()
        cf = self.calculateCostFunction(solution_array)
        feasibility = self.checkFeasibility(solution_array)
        temp_array.append(cf)
        temp_array.append(feasibility)

        sol_array = pd.Series(temp_array)
        sol_array.to_csv(f'{self.file}_res.csv', index=False, header=False)

    """
    Task 5: Implement a method that improves the initially generated solution using hill climbing method which 
    accepts only non-worsening moves. Hill Climb Method :returns 
    """

    def hillClimb(self, n_iterations):

        # Generate an initial solution
        initial_solution = np.array(self.initialSolution())

        # Evaluate current solution
        cost_function = self.calculateCostFunction(self.initialSolution())

        # Track cost functions for plotting
        cost_functions_list = []
        iterations = []

        candidate = initial_solution.copy()

        # run the hill climb
        for i in range(n_iterations):

            # take a step ( Generate a random position for the list )
            random_position = np.random.randint(1, len(candidate))

            # Generate a random number
            random_number = np.random.randint(1, k + 1)

            # Update the random position with random number
            candidate[random_position] = random_number

            # evaluate the cost function at each point.
            candidate_eval = self.calculateCostFunction(candidate)

            # check if we should keep the new point
            if candidate_eval <= cost_function:
                # store the new point
                new_solution = candidate.copy()
                cost_function = candidate_eval
                cost_functions_list.append(cost_function)
                iterations.append(i)

                # report progress
                # print('>%d f(%s) = %.5f' % (i, new_solution, cost_function))

        # Save this solution to csv
        self.write_to_csv(new_solution)
        # Check feasibility and create bar chart.
        self.checkFeasibility(new_solution)
        # Create a plot for the cost functions
        pyplot.title(label=f'Hill Climbing  Method Result')
        pyplot.xlabel('Iterations')
        pyplot.ylabel('Cost')
        pyplot.plot(iterations,cost_functions_list, '--')
        pyplot.show()
        # Generate logs
        self.generateLog(initial_solution, new_solution)
        # Return new solution and cost function
        return [new_solution, cost_function]

    """
    Task 9 : Generate logs
    """

    def generateLog(self, initial_solution, final_solution):
        # Name of the logfile to be saved.
        filename = f'{self.file}_log'
        outfile = open(filename, 'wb')

        # Create a dictionary to store data
        data = {}
        # Add initial solution to it
        data['Initial Solution'] = initial_solution
        # Add cost of that solution
        data['Initial Cost'] = self.calculateCostFunction(initial_solution)
        # Feasibility of the initial solution
        data['Feasibility Of Initial Solution'] = self.checkFeasibility(initial_solution)
        # Add final Solution
        data['Final Solution'] = final_solution
        # Add cost of final solution
        data['Final Cost'] = self.calculateCostFunction(final_solution)
        # Feasibility of the final solution
        data['Feasibility Of Final Solution'] = self.checkFeasibility(final_solution)

        pickle.dump(data, outfile)
        outfile.close()

        return outfile

    def simulated_annealing(self, n_iterations):
        # generate the initial solution
        init_solution = self.initialSolution()
        # Evaluate current solution
        cost_function = self.calculateCostFunction(init_solution)
        cost_functions_list = []
        # Copy initial solution to a candidate
        candidate = init_solution.copy()
        iterations = []

        for i in range(n_iterations):

            # Find out the heaviest and Lightest element
            if i % n_iterations == 0:
                # Heaviest element
                heaviest = init_solution.index(max(init_solution))
                # Lightest Element
                lightest = init_solution.index(min(init_solution))

                # Update the heaviest and lightest elements
                candidate[heaviest], candidate[lightest] = candidate[lightest], candidate[heaviest]
            elif i % n_iterations != 0:
                # take a step ( Generate a random position for the list )
                random_position = np.random.randint(1, len(candidate))

                # Generate a random number
                random_number = np.random.randint(1, k + 1)

                # Update the random position with random number
                candidate[random_position] = random_number

            # evaluate the cost function at each point.
            candidate_eval = self.calculateCostFunction(candidate)

            # check if we should keep the new point
            if candidate_eval <= cost_function or math.exp(-(candidate_eval - cost_function) / 1.4428) > 0.5:
                if self.checkFeasibility(candidate):
                    # 0.5 = 0.721349, 1 = 1.4428
                    # store the new point
                    new_solution = candidate.copy()
                    cost_function = candidate_eval
                    cost_functions_list.append(cost_function)
                    iterations.append(i)

                # report progress
                # print('>%d f(%s) = %.5f' % (i, new_solution, cost_function))

        # Create a plot for the cost functions
        pyplot.title(label=f'Simulated Annealing Method Result')
        pyplot.xlabel('Iterations')
        pyplot.ylabel('Cost')
        pyplot.plot(iterations, cost_functions_list, '--')
        pyplot.show()
        return [new_solution, cost_function]



def main():
    # Introduction
    printIntroduction()

    try:
        runSolver()
    except KeyboardInterrupt:
        return


def printIntroduction():
    print("****************  BALANCING PROBLEM SOLVER  ****************")
    print("Please select the file from the list below")
    print("All CSV files in current directory....")
    # Load all the files in the directory
    i = 0
    for subdir, dirs, files in os.walk('./'):
        for file in files:
            if (file.split())[0][-3:] == 'csv':
                print(f'{i} - {file}')
                i+=1


def runSolver():
    print()
    print("Please enter the name of the csv file")
    print()
    file = input("eg: I1.csv")
    if file.split()[0][-3:] != 'csv':
        print("Not a valid file format.")
        file = ''
    if file!='':
        print("File Loaded Successfully...")
        instance = BalancingProblemSolver(file)
        print("Generating Initial Solution using Randomised Method.")
        initial_solution = instance.initialSolution()
        cost_func_init = instance.calculateCostFunction(initial_solution)
        print(f'Cost function of initial solution is {cost_func_init}')
        print()
        print("Computing using ********Hill Climb Method*************")
        iterations = int(input("Please enter the number of iterations for hill climb. Integers only!"))

        try :
            print("Loading.....")
            sol,cost_func = instance.hillClimb(iterations)
            print(f'Cost of Hill Climb method is {cost_func:0.2f}')
            print("Writing result to csv file")
            instance.write_to_csv(sol)
            print()
            print("Computing using Simulated Annealing Method...")
            iterations_annealing = int(input("Please enter iterations for Annealing method.."))
            solution_sim,cost_sim = instance.simulated_annealing(iterations_annealing)
            print("Cost of simulated annealing is :",cost_sim)


        except Exception as e:
            print(e)






if __name__ == '__main__':
    main()


