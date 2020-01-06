from OutputProfile import *
from enum import Enum
import numpy as np


class ChannelType(Enum):
    leakage_gated = 1
    voltage_gated = 2
    ion_gated = 3


class ChannelPresets(Enum):
    leakage = 1
    fast_potassium = 2
    fast_sodium = 3
    slow_sodium = 4
    hva_calcium = 5
    lva_calcium = 6
    calcium_activated_potassium = 7


class ChannelPreset:
    # cdef ChannelType channel_type
    # cdef str channel_label
    # cdef double equilibrium_potential, m_power, h_power

    def __init__(self, channel_type, channel_label, equilibrium_potential, m_power=0, h_power=0):
        self.channel_type = channel_type
        self.channel_label = channel_label
        self.equilibrium_potential = equilibrium_potential
        self.m_power = m_power
        self.h_power = h_power


channel_presets = {
    ChannelPresets.leakage: ChannelPreset(ChannelType.leakage_gated, "Leakage", -72.57),
    ChannelPresets.fast_potassium: ChannelPreset(ChannelType.voltage_gated, "Fast Potassium", -90.0, m_power=4),
    ChannelPresets.fast_sodium: ChannelPreset(ChannelType.voltage_gated, "Fast Sodium", 50.0, m_power=3, h_power=1),
    ChannelPresets.slow_sodium: ChannelPreset(ChannelType.voltage_gated, "Slow Sodium", 50.0, m_power=3, h_power=1),
    ChannelPresets.hva_calcium: ChannelPreset(ChannelType.voltage_gated, "HVA Calcium", 150.0, m_power=5),
    ChannelPresets.lva_calcium: ChannelPreset(ChannelType.voltage_gated, "LVA Calcium", 150.0, m_power=3, h_power=1),
    ChannelPresets.calcium_activated_potassium: ChannelPreset(ChannelType.ion_gated, "Calcium-Activated Potassium",
                                                              -90.0)
}


class Channel:
    # cdef double equilibrium_potential, max_conductance, delta_t
    # cdef double conductance, current
    # cdef double[] conductance_history, current_history

    def __init__(self, channel_type, channel_label, equilibrium_potential, max_conductance, output_profile=None):
        self.channel_type = channel_type
        self.channel_label = channel_label
        self.equilibrium_potential = equilibrium_potential
        self.max_conductance = max_conductance
        self.output_profile = output_profile
        self.delta_t = 0
        self.compartment_label = ""

        if self.output_profile is None:
            self.output_profile = ChannelOutputProfile()

        self.conductance = 0
        self.current = 0
        self.conductance_history = [self.conductance]
        self.current_history = [self.current]

    def init(self, delta_t, compartment_label):
        self.delta_t = delta_t
        self.compartment_label = compartment_label

    def update_current(self, vm):
        return 0

    def calculate_current(self, vm):
        self.update_current(vm)

        if self.output_profile.keep_data:
            self.conductance_history.append(self.conductance)
            self.current_history.append(self.current)

        return self.current

    def clear_data(self):
        self.conductance = 0
        self.current = 0
        self.conductance_history = [self.conductance]
        self.current_history = [self.current]

    def cleanup(self, directory):
        if self.output_profile.save_plots:
            plot_data(directory,
                      "{} {} Conductance".format(self.compartment_label, self.channel_label), "Conductance (uS)",
                      self.conductance_history, self.delta_t)
            plot_data(directory,
                      "{} {} Current".format(self.compartment_label, self.channel_label), "Current (nA)",
                      self.current_history, self.delta_t)

        self.clear_data()


class LeakageChannel(Channel):
    def __init__(self, max_conductance, channel_label=None, equilibrium_potential=0.0,
                 channel_preset=None, output_profile=None):
        if channel_preset is not None:
            channel_label = channel_label if channel_label else channel_presets[channel_preset].channel_label
            equilibrium_potential = equilibrium_potential if equilibrium_potential else \
                channel_presets[channel_preset].equilibrium_potential
        super().__init__(ChannelType.leakage_gated, channel_label, equilibrium_potential, max_conductance, output_profile)

        self.conductance = max_conductance

    def update_current(self, vm):
        self.current = self.max_conductance * (vm - self.equilibrium_potential)


class VoltageGatedChannel(Channel):
    # cdef int m_power, h_power
    # # cdef double[] activation_parameters, inactivation_parameters
    # cdef double m, h, previous_vm

    def __init__(self, max_conductance, m_power=0, h_power=0, channel_label=None, equilibrium_potential=0.0,
                 activation_parameters=None, inactivation_parameters=None, channel_preset=None, output_profile=None):
        if channel_preset is not None:
            channel_label = channel_label if channel_label else channel_presets[channel_preset].channel_label
            equilibrium_potential = equilibrium_potential if equilibrium_potential else \
                channel_presets[channel_preset].equilibrium_potential
        super().__init__(ChannelType.voltage_gated, channel_label, equilibrium_potential, max_conductance, output_profile)

        self.m_power = m_power if m_power else channel_presets[channel_preset].m_power
        self.h_power = h_power if h_power else channel_presets[channel_preset].h_power

        self.activation_parameters = activation_parameters
        self.inactivation_parameters = inactivation_parameters

        self.m = -1
        self.h = -1
        self.previous_vm = -72.57

    def calculate_activation_alpha(self, vm):
        if vm - self.activation_parameters[4] == 0.0:
            return 0.0
        cdef double res = self.activation_parameters[0] * (vm - self.activation_parameters[1]) / \
               (1 - np.exp((self.activation_parameters[1] - vm) / self.activation_parameters[2]))

        return res

    def calculate_activation_beta(self, vm):
        if vm - self.activation_parameters[4] == 0.0:
            return 0.0
        cdef double res = self.activation_parameters[3] * (self.activation_parameters[4] - vm) / \
               (1 - np.exp((vm - self.activation_parameters[4]) / self.activation_parameters[5]))

        return res

    def calculate_inactivation_alpha(self, vm):
        if vm - self.inactivation_parameters[1] == 0.0:
            return 0.0
        cdef double res = self.inactivation_parameters[0] * (self.inactivation_parameters[1] - vm) / \
               (1 - np.exp((vm - self.inactivation_parameters[1]) / self.inactivation_parameters[2]))

        return res

    def calculate_inactivation_beta(self, vm):
        if vm - self.inactivation_parameters[4] == 0.0:
            return 0.0
        cdef double res = self.inactivation_parameters[3] / \
               (1 + np.exp((self.inactivation_parameters[4] - vm) / self.inactivation_parameters[5]))

        return res

    def update_m(self, vm):
        cdef double alpha = self.calculate_activation_alpha(vm)
        cdef double beta = self.calculate_activation_beta(vm)
        cdef double alpha_beta_sum = alpha + beta

        if self.m == -1:
            self.m = alpha / (alpha + beta)
        else:
            self.m = (1 - self.m) * (alpha - alpha * np.exp(-1 * alpha_beta_sum * self.delta_t)) / alpha_beta_sum + \
                     self.m * (alpha + beta * np.exp(-1 * alpha_beta_sum * self.delta_t)) / alpha_beta_sum

    def update_h(self, vm):
        cdef double alpha = self.calculate_inactivation_alpha(vm)
        cdef double beta = self.calculate_inactivation_beta(vm)
        cdef double alpha_beta_sum = alpha + beta

        if self.h == -1:
            self.h = alpha / (alpha + beta)
        else:
            self.h = (1 - self.h) * (alpha - alpha * np.exp(-1 * alpha_beta_sum * self.delta_t)) / alpha_beta_sum + \
                     self.h * (alpha + beta * np.exp(-1 * alpha_beta_sum * self.delta_t)) / alpha_beta_sum

    def update_current(self, vm):
        self.update_m(self.previous_vm)
        self.conductance = self.max_conductance * np.power(self.m, self.m_power)
        if self.h_power is not 0:
            self.update_h(self.previous_vm)
            self.conductance *= np.power(self.h, self.h_power)

        self.current = self.conductance * (vm - self.equilibrium_potential)
        self.previous_vm = vm

    def calculate_activation_alpha_vec(self, vm):
        return np.nan_to_num(self.activation_parameters[0] * np.subtract(vm, self.activation_parameters[1]) / \
               np.subtract(1, np.exp(np.subtract(self.activation_parameters[1], vm) / self.activation_parameters[2])))

    def calculate_activation_beta_vec(self, vm):
        return np.nan_to_num(self.activation_parameters[3] * np.subtract(self.activation_parameters[4], vm) / \
               np.subtract(1, np.exp(np.subtract(vm, self.activation_parameters[4]) / self.activation_parameters[5])))

    def calculate_inactivation_alpha_vec(self, vm):
        return np.nan_to_num(self.inactivation_parameters[0] * np.subtract(self.inactivation_parameters[1], vm) / \
               np.subtract(1, np.exp(np.subtract(vm, self.inactivation_parameters[1]) / self.inactivation_parameters[2])))

    def calculate_inactivation_beta_vec(self, vm):
        return np.nan_to_num(self.inactivation_parameters[3] / \
               np.add(1, np.exp(np.subtract(self.inactivation_parameters[4], vm) / self.inactivation_parameters[5])))

class IonGatedChannel(Channel):
    # cdef double decay, ion_pool

    def __init__(self, max_conductance, decay, channel_label=None, equilibrium_potential=0.0,
                 activator_channels=None, channel_preset=None, output_profile=None):
        if channel_preset is not None:
            channel_label = channel_label if channel_label else channel_presets[channel_preset].channel_label
            equilibrium_potential = equilibrium_potential if equilibrium_potential else \
                channel_presets[channel_preset].equilibrium_potential
        super().__init__(ChannelType.ion_gated, channel_label, equilibrium_potential, max_conductance, output_profile)

        self.decay = decay
        self.activator_channels = activator_channels

        self.ion_pool = 0

    def update_ion_pool(self, vm):
        cdef double accumulation = 0

        for (channel, channel_accumulation) in self.activator_channels:
            accumulation += channel_accumulation * abs(channel.current) / channel.max_conductance

        self.ion_pool = max(self.ion_pool + (accumulation - self.decay * self.ion_pool) * self.delta_t, 0.0)

    def update_current(self, vm):
        self.update_ion_pool(vm)
        self.conductance = self.max_conductance * self.ion_pool
        self.current = self.conductance * (vm - self.equilibrium_potential)