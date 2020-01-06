
class Model:
    def __init__(self, delta_t, compartments=None, coupling_conductances=None):
        self.delta_t = delta_t
        self.compartments = compartments
        self.coupling_conductances = coupling_conductances

        for compartment in self.compartments:
            compartment.init(delta_t)

    def run_simulation(self, duration, stimulus_current, stimulus_start, stimulus_duration):
        compartment_vms = []
        for compartment in self.compartments:
            compartment_vms.append(compartment.initial_vm)
        coupling_currents = [0] * len(self.coupling_conductances)

        cdef double time = 0.0
        cdef double stimulus = 0.0
        cdef double coupling_current = 0.0
        while time < duration:
            stimulus = stimulus_current if stimulus_start <= time < (stimulus_start + stimulus_duration) else 0.0

            for i in range(len(coupling_currents)):
                coupling_currents[i] = self.coupling_conductances[i] * (compartment_vms[i] - compartment_vms[i + 1])

            for i, compartment in enumerate(self.compartments):
                coupling_current = (0.0 if i == 0 else (-1 * coupling_currents[i - 1])) + \
                                   (0.0 if i == len(self.coupling_conductances) else coupling_currents[i])
                compartment_vms[i] = compartment.calculate_vm(stimulus if compartment.stimulus_compartment else 0.0,
                                                              coupling_current)

            time += self.delta_t

    def cleanup(self, directory):
        for compartment in self.compartments:
            compartment.cleanup(directory)

        # with open(os.path.join(directory, "model.pickle"), 'wb') as handle:
        #     pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)