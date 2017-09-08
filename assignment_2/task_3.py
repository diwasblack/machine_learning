import os
import math
import random
import logging

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def htheta_function(x, theta_0=0, theta_1=0):
    """
    Hypothesis function to use
    """

    z = theta_0 + theta_1 * x

    result = 1.0 / (1 + pow(math.e, -z))

    # Return value of sigmoid function
    return result


def obtain_points(func, theta_0, theta_1, min_x, max_x, step=0.1):
    """
    Return a tuple of x and the corresponding points for the given x
    """

    x_values = []
    y_values = []

    x = min_x

    while x <= max_x:

        y_values.append(func(x, theta_0, theta_1))
        x_values.append(x)

        x += step

    return (x_values, y_values)


def main():
    """
    Entry point for program
    """

    training_inputs = [
        0.5, 0.7, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5,
        4, 4.25, 4.5, 4.75, 5, 5.5
    ]
    training_outputs = [
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1
    ]

    # Assign random values to theta parameters
    theta_0 = random.random()
    theta_1 = random.random()

    # Learning rate
    alpha = 0.01

    output_dir = "output_3"
    max_no_of_iterations = 10000

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

    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))

    no_of_iterations = 0
    stop_training = False
    while not (stop_training):

        # Temporary variables to hold data
        theta_0_correction = 0
        theta_1_correction = 0

        for training_sample in zipped_training_samples:
            # Unpack training input output
            training_input, training_ouput = training_sample

            htheta_value = htheta_function(training_input, theta_0, theta_1)

            # Calculate error
            error = 100 * htheta_value * (1 - htheta_value) * (
                1.0 - 2 * training_ouput)

            theta_0_correction += error * 1
            theta_1_correction += error * training_input

        # Increment parameters
        theta_0 -= (alpha / no_of_training_sample) * theta_0_correction
        theta_1 -= (alpha / no_of_training_sample) * theta_1_correction

        no_of_iterations += 1

        if no_of_iterations >= max_no_of_iterations:
            logging.info("Reached maximum number of iterations")
            break

    logging.info(
        "Final values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    equation_points = obtain_points(htheta_function, theta_0, theta_1, 0, 6)

    plt.plot(equation_points[0], equation_points[1])

    plt.savefig(
        os.path.join(output_dir, "regression_curve_{}.png".format(
            str(alpha).replace(".", "_"))))


if __name__ == "__main__":
    main()
