from enum import Enum
import numpy as np


class ConstraintType(Enum):
    relative_maximum_conductance = 1
    b_parameter_range = 2
    activation_voltage_range = 3
    inactivation_voltage_range = 4
    activation_inactivation_intersection = 5
    internal_time_constants = 6
    relative_activation_time_constants = 7
    relative_inactivation_time_constants = 8


class ParameterConstraint:
    def __init__(self, constraint_type, channel_one, channel_two=None, v_rest=None,
                 activation_range=None, rest_probability=None):
        self.constraint_type = constraint_type

        if self.constraint_type is ConstraintType.relative_maximum_conductance:
            self.channel_one = channel_one
            self.channel_two = channel_two
        elif self.constraint_type is ConstraintType.b_parameter_range:
            self.channel_one = channel_one
            self.v_rest = v_rest
        elif self.constraint_type is ConstraintType.activation_voltage_range:
            self.channel_one = channel_one
            self.v_rest = v_rest
            self.activation_range = activation_range
            self.rest_probability = rest_probability
        elif self.constraint_type is ConstraintType.inactivation_voltage_range:
            self.channel_one = channel_one
            self.v_rest = v_rest
            self.activation_range = activation_range
            self.rest_probability = rest_probability
        elif self.constraint_type is ConstraintType.activation_inactivation_intersection:
            self.channel_one = channel_one
        elif self.constraint_type is ConstraintType.internal_time_constants:
            self.channel_one = channel_one
        elif self.constraint_type is ConstraintType.relative_activation_time_constants:
            self.channel_one = channel_one
            self.channel_two = channel_two
        elif self.constraint_type is ConstraintType.relative_inactivation_time_constants:
            self.channel_one = channel_one
            self.channel_two = channel_two

    def evaluate_constraint(self, print_evaluation_details=False):
        constraint_pass = False
        if self.constraint_type is ConstraintType.relative_maximum_conductance:
            constraint_pass = self.channel_one.max_conductance > self.channel_two.max_conductance
        elif self.constraint_type is ConstraintType.b_parameter_range:
            constraint_pass = self.b_parameter_constraint()
        elif self.constraint_type is ConstraintType.activation_voltage_range:
            constraint_pass = self.gate_voltage_constraint()
        elif self.constraint_type is ConstraintType.inactivation_voltage_range:
            constraint_pass = self.gate_voltage_constraint()
        elif self.constraint_type is ConstraintType.activation_inactivation_intersection:
            constraint_pass = self.gate_intersection_constraint()
        elif self.constraint_type is ConstraintType.internal_time_constants:
            constraint_pass = self.internal_time_constant_constraint()
        elif self.constraint_type is ConstraintType.relative_activation_time_constants:
            constraint_pass = self.relative_time_constant_constraint()
        elif self.constraint_type is ConstraintType.relative_inactivation_time_constants:
            constraint_pass = self.relative_time_constant_constraint()

        if print_evaluation_details:
            print("{} - {}: {}".format(self.channel_one.channel_label, self.constraint_type, constraint_pass))
        return constraint_pass

    def b_parameter_constraint(self):
        cdef double b_range_low = self.v_rest
        cdef double b_range_high = 0

        activation_alpha = b_range_low < self.channel_one.activation_parameters[1] < b_range_high
        activation_beta = b_range_low < self.channel_one.activation_parameters[4] < b_range_high
        constraint_pass = activation_alpha and activation_beta

        if self.channel_one.inactivation_parameters is not None:
            inactivation_alpha = b_range_low < self.channel_one.inactivation_parameters[1] < b_range_high
            inactivation_beta = b_range_low < self.channel_one.inactivation_parameters[4] < b_range_high
            constraint_pass = constraint_pass and inactivation_alpha and inactivation_beta

        return constraint_pass

    def gate_voltage_constraint(self):
        calculate_alpha = None
        calculate_beta = None

        if self.constraint_type is ConstraintType.activation_voltage_range:
            calculate_alpha = self.channel_one.calculate_activation_alpha_vec
            calculate_beta = self.channel_one.calculate_activation_beta_vec
        elif self.constraint_type is ConstraintType.inactivation_voltage_range:
            calculate_alpha = self.channel_one.calculate_inactivation_alpha_vec
            calculate_beta = self.channel_one.calculate_inactivation_beta_vec

        vm_sample = np.add(np.arange(-100, 101, 1), 0.0001)
        alpha_sample = calculate_alpha(vm_sample)
        beta_sample = calculate_beta(vm_sample)
        probability_sample = alpha_sample / np.add(alpha_sample, beta_sample)

        a = False
        cdef double activation_threshold
        if self.constraint_type is ConstraintType.activation_voltage_range:
            activation_threshold = vm_sample[np.searchsorted(probability_sample, 0.2)]
            a = self.activation_range[0] < activation_threshold < self.activation_range[1]
        elif self.constraint_type is ConstraintType.inactivation_voltage_range:
            activation_threshold = vm_sample[len(probability_sample) - np.searchsorted(np.flip(probability_sample), 0.2)]
            a = self.activation_range[0] < activation_threshold < self.activation_range[1]

        cdef double rest_potential = 0 if self.constraint_type is ConstraintType.activation_voltage_range else self.v_rest
        b = probability_sample[np.searchsorted(vm_sample, rest_potential)] > self.rest_probability

        return a and b

    def gate_intersection_constraint(self):
        cdef double intersection_probability_low = 0.05
        cdef double intersection_probability_high = 0.3
        vm_sample = np.add(np.arange(-100, 101, 1), 0.0001)

        activation_alpha_sample = self.channel_one.calculate_activation_alpha_vec(vm_sample)
        activation_beta_sample = self.channel_one.calculate_activation_beta_vec(vm_sample)
        activation_probability_sample = activation_alpha_sample / np.add(activation_alpha_sample,
                                                                         activation_beta_sample)

        inactivation_alpha_sample = self.channel_one.calculate_inactivation_alpha_vec(vm_sample)
        inactivation_beta_sample = self.channel_one.calculate_inactivation_beta_vec(vm_sample)
        inactivation_probability_sample = inactivation_alpha_sample / np.add(inactivation_alpha_sample,
                                                                             inactivation_beta_sample)

        cdef double intersection_probability = activation_probability_sample[np.argmin(np.abs(np.subtract(
            activation_probability_sample, inactivation_probability_sample)))]

        return intersection_probability_low < intersection_probability < intersection_probability_high

    def internal_time_constant_constraint(self):
        vm_sample = np.add(np.arange(-100, 101, 1), 0.0001)

        activation_alpha_sample = self.channel_one.calculate_activation_alpha_vec(vm_sample)
        activation_beta_sample = self.channel_one.calculate_activation_beta_vec(vm_sample)
        activation_time_constant = 1 / np.add(activation_alpha_sample, activation_beta_sample)

        inactivation_alpha_sample = self.channel_one.calculate_inactivation_alpha_vec(vm_sample)
        inactivation_beta_sample = self.channel_one.calculate_inactivation_beta_vec(vm_sample)
        inactivation_time_constant = 1 / np.add(inactivation_alpha_sample, inactivation_beta_sample)

        return np.mean(np.subtract(activation_time_constant, inactivation_time_constant)) < 0

    def relative_time_constant_constraint(self):
        vm_sample = np.add(np.arange(-100, 101, 1), 0.0001)

        channel_one_time_constant = None
        channel_two_time_constant = None
        if self.constraint_type is ConstraintType.relative_activation_time_constants:
            channel_one_alpha_sample = self.channel_one.calculate_activation_alpha_vec(vm_sample)
            channel_one_beta_sample = self.channel_one.calculate_activation_beta_vec(vm_sample)
            channel_one_time_constant = 1 / np.add(channel_one_alpha_sample, channel_one_beta_sample)

            channel_two_alpha_sample = self.channel_two.calculate_activation_alpha_vec(vm_sample)
            channel_two_beta_sample = self.channel_two.calculate_activation_beta_vec(vm_sample)
            channel_two_time_constant = 1 / np.add(channel_two_alpha_sample, channel_two_beta_sample)
        elif self.constraint_type is ConstraintType.relative_inactivation_time_constants:
            channel_one_alpha_sample = self.channel_one.calculate_inactivation_alpha_vec(vm_sample)
            channel_one_beta_sample = self.channel_one.calculate_inactivation_beta_vec(vm_sample)
            channel_one_time_constant = 1 / np.add(channel_one_alpha_sample, channel_one_beta_sample)

            channel_two_alpha_sample = self.channel_two.calculate_inactivation_alpha_vec(vm_sample)
            channel_two_beta_sample = self.channel_two.calculate_inactivation_beta_vec(vm_sample)
            channel_two_time_constant = 1 / np.add(channel_two_alpha_sample, channel_two_beta_sample)

        print("C1 t:", np.mean(channel_one_time_constant))
        print("C2 t:", np.mean(channel_two_time_constant))

        return np.mean(np.subtract(channel_one_time_constant, channel_two_time_constant)) < 0