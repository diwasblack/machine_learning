import os
import random
import logging

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def htheta_function(x, theta_0, theta_1):
    """
    Linear function that is to optimized
    """
    return theta_0 + theta_1 * x


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


def make_classifier(theta_0, theta_1, threshold):
    """
    Construct function to perform the predition
    """

    def predict_result(number_of_hours):
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

    output_dir = "output_1"
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

        logging.debug("Iteration:{} Cost:{}".format(no_of_iterations,
                                                    mean_squared_error))

        # Increment parameters
        theta_0 -= (alpha / no_of_training_sample) * theta_0_correction
        theta_1 -= (alpha / no_of_training_sample) * theta_1_correction

        no_of_iterations += 1

        if no_of_iterations >= max_no_of_iterations:
            logging.info("Reached maximum number of iterations")
            break

    logging.info(
        "Final values\ntheta0: {}\ntheta1: {}".format(theta_0, theta_1))

    # Plot the obtained line
    line_points = obtain_points(htheta_function, theta_0, theta_1, 0, 6)

    # Overlap regression line plot with scatter plot
    # plt.clf()

    plt.plot(line_points[0], line_points[1])

    plt.savefig(
        os.path.join(output_dir, "regression_line_{}.png".format(
            str(alpha).replace(".", "_"))))

    # Clear previous plot
    plt.clf()

    # Construct classifier
    classifier = make_classifier(theta_0, theta_1, 0.5)

    predict_x = 5.0

    # Try to predict the result for value
    logging.info("The predicted result for x {} is:\nClass: {}".format(
        predict_x, classifier(predict_x)))


if __name__ == "__main__":
    main()
