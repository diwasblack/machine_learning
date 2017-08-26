import os
import random
import logging
import statistics
from collections import deque

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
output_dir = "output_1"


def htheta_function(x, theta_0, theta_1):
    """
    Linear function that is to optimized
    """
    return theta_0 + theta_1 * x


def ymxc_line_points(m, c, min_x, max_x, step=1):
    """
    Function to return the points for y = mx + c
    """

    x_values = []
    y_values = []

    x = min_x

    while x <= max_x:
        y_value = m * x + c
        x_values.append(x)
        y_values.append(y_value)

        x += step

    return (x_values, y_values)


def main():
    """
    Entry point for program
    """

    training_inputs = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    training_outputs = [5.1, 6.1, 6.9, 7.8, 9.2, 9.9, 11.5, 12, 12.8]

    # Assign random values to theta parameters
    theta_0 = random.random()
    theta_1 = random.random()

    # Minimum threshold after which the gradient descent should stop
    threshold = 0.01

    # Learning rate
    alpha = 0.01

    # Deque to hold value of cost function for the last 10 iterations
    last_cost_values = deque(maxlen=10)

    # Number of iterations to plot for cost function
    number_of_iterations_to_plot = 500

    # History of cost values
    cost_values = list()

    if not (os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    logging.info(
        "Initial values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    zipped_training_samples = [
        list(x) for x in zip(training_inputs, training_outputs)
    ]
    no_of_training_sample = len(zipped_training_samples)

    plt.scatter(training_inputs, training_outputs)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter plot of training data")

    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))

    no_of_iterations = 0
    stop_training = False
    while not (stop_training):

        # Temporary variables to hold data
        total_squared_error = 0
        theta_0_correction = 0
        theta_1_correction = 0

        for training_sample in zipped_training_samples:

            # Unpack training input output
            training_input, training_ouput = training_sample

            error = htheta_function(training_input, theta_0,
                                    theta_1) - training_ouput

            # Calculate squared error for training samples
            try:
                squared_error = pow(error, 2.0)
            except OverflowError:
                logging.info("OverflowError occured")
                stop_training = True

                # Replace squared error with an estimate
                squared_error = abs(error)

            theta_0_correction += error * 1
            theta_1_correction += error * training_input

            total_squared_error += squared_error

        # Calculate the mean squared error
        mean_squared_error = (1.0 / (
            2.0 * no_of_training_sample)) * total_squared_error

        # Add cost value to history
        cost_values.append(mean_squared_error)

        last_cost_values.append(mean_squared_error)

        # Calculate the average cost value
        last_cost_average = statistics.mean(last_cost_values)

        logging.debug("Iteration:{} Cost:{}, Average:{}".format(
            no_of_iterations, mean_squared_error, last_cost_average))

        # Increment parameters
        theta_0 -= (alpha / no_of_training_sample) * theta_0_correction
        theta_1 -= (alpha / no_of_training_sample) * theta_1_correction

        no_of_iterations += 1

        if last_cost_average <= threshold:
            logging.info("Reached minimum threshold")
            break

        if no_of_iterations >= 10000:
            logging.info("Reached maximum number of iterations")
            break

    logging.info(
        "Final values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    # Plot the obtained line
    line_points = ymxc_line_points(theta_1, theta_0, 0, 10, 0.1)

    # Overlap regression line plot with scatter plot
    # plt.clf()

    plt.plot(line_points[0], line_points[1])
    plt.title("Regression line")

    plt.savefig(
        os.path.join(output_dir, "regression_line_{}.png".format(
            str(alpha).replace(".", "_"))))

    # Clear previous plot
    plt.clf()

    minimun_points_to_plot = min(number_of_iterations_to_plot,
                                 no_of_iterations)

    plt.plot(
        range(1, minimun_points_to_plot + 1),
        cost_values[:minimun_points_to_plot])

    plt.xlabel("Iteration")
    plt.ylabel("J(θ)")
    plt.title("Mean squared error per iteration")

    plt.savefig(
        os.path.join(output_dir, "cost_function_alpha_{}.png".format(
            str(alpha).replace(".", "_"))))


if __name__ == "__main__":
    main()
