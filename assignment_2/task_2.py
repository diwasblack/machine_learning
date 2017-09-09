import math
import random
import logging

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def htheta_function(x, theta_0, theta_1):
    """
    Sidmoid function to use for classification
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


def make_classifier(theta_0, theta_1):
    """
    Construct function to perform the predition
    """

    def predict_result(number_of_hours, threshold=0.5):
        # Obtain result for the htheta function
        htheta_value = htheta_function(number_of_hours, theta_0, theta_1)

        if htheta_value >= threshold:
            return 1
        return 0

    # Return the constructed function
    return predict_result


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

    max_no_of_iterations = 10000

    logging.info(
        "Initial values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    zipped_training_samples = [
        list(x) for x in zip(training_inputs, training_outputs)
    ]
    no_of_training_sample = len(zipped_training_samples)

    plt.scatter(training_inputs, training_outputs)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.savefig("scatter_plot.png")

    no_of_iterations = 0
    stop_training = False
    while not (stop_training):

        total_error_signal = 0
        theta_0_correction = 0
        theta_1_correction = 0

        for training_sample in zipped_training_samples:
            # Unpack training input output
            training_input, training_ouput = training_sample

            htheta_value = htheta_function(training_input, theta_0, theta_1)

            # Calculate individual error for training samples
            training_cost = training_ouput * math.log(htheta_value) + (
                1.0 - training_ouput) * math.log(1.0 - htheta_value)

            error_signal = htheta_value - training_ouput

            theta_0_correction += error_signal * 1
            theta_1_correction += error_signal * training_input

            total_error_signal += training_cost

        total_cost = (-1.0 / no_of_training_sample) * total_error_signal

        logging.debug(
            "Iteration:{} Cost:{}".format(no_of_iterations, total_cost))

        # Increment parameters
        theta_0 -= (alpha / no_of_training_sample) * theta_0_correction
        theta_1 -= (alpha / no_of_training_sample) * theta_1_correction

        no_of_iterations += 1

        if no_of_iterations >= max_no_of_iterations:
            logging.info("Reached maximum number of iterations")
            break

    logging.info(
        "Final values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    # Overlap regression line plot with scatter plot
    # plt.clf()

    equation_points = obtain_points(htheta_function, theta_0, theta_1, 0, 6)

    plt.plot(equation_points[0], equation_points[1])

    plt.savefig("logistic_regression_curve_{}.png".format(
        str(alpha).replace(".", "_")))

    classifier = make_classifier(theta_0, theta_1)

    predict_x = 2.0

    # Try to predict the result for value
    logging.info("The predicted result for x {} is:\nClass: {}".format(
        predict_x, classifier(predict_x)))


if __name__ == "__main__":
    main()
