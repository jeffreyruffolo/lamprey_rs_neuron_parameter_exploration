from Channel import *
from OutputProfile import *

class Compartment():
    def __init__(self, compartment_label, capacitance, initial_vm, channels=None, stimulus_compartment=False,
                 output_profile=None):
        self.compartment_label = compartment_label
        self.capacitance = capacitance
        self.initial_vm = initial_vm
        self.vm = initial_vm
        self.stimulus_compartment = stimulus_compartment
        self.output_profile = output_profile
        self.delta_t = 0
        self.vm_history = [initial_vm]

        if self.output_profile is None:
            self.output_profile = CompartmentOutputProfile()

        leakage_channels = []
        voltage_gated_channels = []
        ion_gated_channels = []
        for channel in channels:
            if channel.channel_type is ChannelType.leakage_gated:
                leakage_channels.append(channel)
            elif channel.channel_type is ChannelType.voltage_gated:
                leakage_channels.append(channel)
            elif channel.channel_type is ChannelType.ion_gated:
                ion_gated_channels.append(channel)
        self.channels = leakage_channels + voltage_gated_channels + ion_gated_channels

    def init(self, delta_t):
        for channel in self.channels:
            channel.init(delta_t, self.compartment_label)
        self.delta_t = delta_t

    def calculate_vm(self, stimulus_current, coupling_current):
        cdef double channel_current = 0.0
        for channel in self.channels:
            channel_current += channel.calculate_current(self.vm)

        self.vm += (stimulus_current - channel_current - coupling_current) / self.capacitance * self.delta_t

        if self.output_profile.keep_data:
            self.vm_history.append(self.vm)

        return self.vm

    def clear_data(self):
        for channel in self.channels:
            channel.clear_data()
        self.vm = self.initial_vm
        self.vm_history = [self.initial_vm]

    def cleanup(self, directory):
        for channel in self.channels:
            channel.cleanup(directory)

        if self.output_profile.save_plots:
            plot_data(directory, "{} Vm".format(self.compartment_label), "Vm (mV)",
                      self.vm_history, self.delta_t)

            dvdt = np.subtract(self.vm_history[:-1], self.vm_history[1:]) / self.delta_t
            plot_axes(directory, "{} dvdt".format(self.compartment_label), x_label="Vm (mV)", y_label="dv/dt",
                      x_data=self.vm_history[1:], y_data=dvdt)

            np.savetxt(os.path.join(directory, "{}_vm.csv".format(self.compartment_label)),
                                    np.column_stack((np.arange(0, len(self.vm_history) * self.delta_t, self.delta_t), np.asarray(self.vm_history))),
                                    delimiter=',')
            np.savetxt(os.path.join(directory, "{}_dvdt.csv".format(self.compartment_label)),
                                    np.column_stack((np.asarray(self.vm_history[1:]), dvdt)),
                                    delimiter=',')

        self.clear_data()