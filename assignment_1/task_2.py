import os
import random
import logging

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def htheta_function(x1, x2, theta_0, theta_1, theta_2):
    """
    Linear function that is to optimized
    """
    return theta_0 + theta_1 * x1 + theta_2 * x2


def obtain_range(list_object):
    """
    Return the range of the list
    """

    min_value = min(list_object)
    max_value = max(list_object)

    return max_value - min_value


def scale_training_data(training_data):
    """
    Normalize/scale training data using some technique
    """

    # Split training data column-wise
    training_x1, training_x2 = [list(i) for i in zip(*training_data)]

    # Scale x1 by using range
    x1_range = obtain_range(training_x1)

    logging.info("scaling factor for x1: {}".format(x1_range))
    scaled_x1 = [x1 / x1_range for x1 in training_x1]

    # Scale x2 by using range
    x2_range = obtain_range(training_x2)
    logging.info("scaling factor for x2: {}".format(x2_range))
    scaled_x2 = [x2 / x2_range for x2 in training_x2]

    scaled_training_data = [tuple(x) for x in zip(scaled_x1, scaled_x2)]
    return scaled_training_data, x1_range, x2_range


def main():
    """
    Entry point for program
    """

    training_inputs = [(2, 70), (3, 30), (4, 80), (4, 20), (3, 50), (7, 10),
                       (5, 50), (3, 90), (2, 20)]

    training_inputs, x1_scale, x2_scale = scale_training_data(training_inputs)

    training_outputs = [79.4, 41.5, 97.5, 36.1, 63.2, 39.5, 69.8, 103.5, 29.5]

    # Assign random values to theta parameters
    theta_0 = random.random()
    theta_1 = random.random()
    theta_2 = random.random()

    # Learning rate
    alpha = 0.01

    # Number of iterations to plot for cost function
    number_of_iterations_to_plot = 500

    # History of cost values
    cost_values = list()

    output_dir = "output_2"
    max_no_of_iterations = 10000

    if not (os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    logging.info("Initial values\ntheta0: {}\ntheta1: {}\ntheta2: {}".format(
        theta_0, theta_1, theta_2))

    zipped_training_samples = [
        list(x) for x in zip(training_inputs, training_outputs)
    ]
    no_of_training_sample = len(zipped_training_samples)

    no_of_iterations = 0
    stop_training = False
    while not (stop_training):

        # Temporary variables to hold data
        total_squared_error = 0
        theta_0_correction = 0
        theta_1_correction = 0
        theta_2_correction = 0

        for training_sample in zipped_training_samples:

            # Unpack training input output
            training_input, training_ouput = training_sample

            error = htheta_function(training_input[0], training_input[1],
                                    theta_0, theta_1, theta_2) - training_ouput

            # Calculate squared error for training samples
            try:
                squared_error = pow(error, 2.0)
            except OverflowError:
                logging.info("OverflowError occured")
                stop_training = True

                # Replace squared error with an estimate
                squared_error = abs(error)

            theta_0_correction += error * 1
            theta_1_correction += error * training_input[0]
            theta_2_correction += error * training_input[1]

            total_squared_error += squared_error

        # Calculate the mean squared error
        mean_squared_error = (1.0 / (
            2.0 * no_of_training_sample)) * total_squared_error

        # Add cost value to history
        cost_values.append(mean_squared_error)

        logging.debug("Iteration:{} Cost:{}".format(no_of_iterations,
                                                    mean_squared_error))

        # Increment parameters
        theta_0 -= (alpha / no_of_training_sample) * theta_0_correction
        theta_1 -= (alpha / no_of_training_sample) * theta_1_correction
        theta_2 -= (alpha / no_of_training_sample) * theta_2_correction

        if no_of_iterations >= max_no_of_iterations:
            logging.info("Reached maximum number of iterations")
            break

        no_of_iterations += 1

    logging.info("Final values\ntheta0: {}\ntheta1: {}\ntheta2: {}".format(
        theta_0, theta_1 / x1_scale, theta_2 / x2_scale))

    minimum_points_to_plot = min(number_of_iterations_to_plot,
                                 no_of_iterations)

    plt.plot(
        range(1, minimum_points_to_plot + 1),
        cost_values[:minimum_points_to_plot])

    plt.xlabel("Iteration")
    plt.ylabel("J(θ)")
    plt.savefig(
        os.path.join(output_dir, "cost_function_alpha_{}.png".format(
            str(alpha).replace(".", "_"))))


if __name__ == "__main__":
    main()
