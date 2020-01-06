import math
from scipy.signal import find_peaks
import datetime

from Compartment import *
from Model import *
from OutputProfile import *
from ParameterConstraint import *
from test_parameters import parameters60, p1


# feature evaluation parameters
cdef double allowed_sds = 2.0
cdef double detection_sds = 3.0

# single AP values
cdef double _vrest_vm = -72.57
cdef double _vrest_vm_sd = 4.33

cdef double _ap_vm = 103.86
cdef double _ap_vm_sd = 7.22

cdef double _fAHP_vm = -4.57
cdef double _fAHP_vm_sd = 2.92
cdef double _fAHP_delay = 1.95
cdef double _fAHP_delay_sd = 0.42

cdef double _ADP_vm = 2.88
cdef double _ADP_vm_sd = 1.98
cdef double _ADP_delay = 6.95
cdef double _ADP_delay_sd = 2.52

cdef double _sAHP_vm = -1.49
cdef double _sAHP_vm_sd = 1.24
cdef double _sAHP_delay = 38.56
cdef double _sAHP_delay_sd = 13.44

cdef double _dvdt_rise = 260.32
cdef double _dvdt_rise_sd = 34.25
cdef double _dvdt_fall = -138.95
cdef double _dvdt_fall_sd = 27.02

# repetitive firing values
cdef double _initial_freq_min = 10.0
cdef double _initial_freq_max = 30.0
cdef double _sfa_min = 0.20
cdef double _sfa_max = 0.80

# evaluation parameters
cdef double _s_duration = 100.0
cdef double _s_delta_t = 0.005
cdef double _s_stimulus = 10.0
cdef double _s_stimulus_start = 2.0
cdef double _s_stimulus_duration_min = 0.5
cdef double _s_stimulus_duration_iterator = 0.01
cdef double _s_stimulus_duration_max = 3.0

cdef double _r_duration = 1000.0
cdef double _r_delta_t = 0.01
cdef double _r_stimulus_min = 0.5
cdef double _r_stimulus_iterator = 0.05
cdef double _r_stimulus_max = 4.0
cdef double _r_stimulus_start = 10.0
cdef double _r_stimulus_duration = 980.0


class ModelType(Enum):
    full_model = 1
    soma_conductance = 2
    soma_kna_conductance = 3


class RepetitiveFiringOutcome(Enum):
    success = 1
    low_firing = 2
    erratic_firing = 3
    low_initial_freq = 4
    high_initial_freq = 5
    high_sfa = 6


def test_parameter_fitness(parameters, model_type=ModelType.soma_kna_conductance, use_constraints=True):
    model_runner = ModelRunner(model_type, variable_parameters=parameters,
                           initial_vm=-72.57, print_evaluation_details=False, use_constraints=use_constraints)
    return model_runner.evaluate_model()


def test_parameter_constraints(parameters, model_type=ModelType.soma_kna_conductance):
    model_runner = ModelRunner(model_type, variable_parameters=parameters,
                           initial_vm=-72.57, print_evaluation_details=False)
    return model_runner.constraint_pass


def get_parameter_scores(parameters, model_type=ModelType.soma_kna_conductance, use_constraints=True):
    model_runner = ModelRunner(model_type, variable_parameters=parameters,
                           initial_vm=-72.57, print_evaluation_details=False, use_constraints=use_constraints)
    return model_runner.evaluate_model(score_breakdown=True)[1]


class ModelRunner:
    def __init__(self, model_type, variable_parameters, initial_vm, print_evaluation_details, use_constraints=True):
        self.model_type = model_type
        self.variable_parameters = variable_parameters
        self.active_parameters = self.get_active_parameters()
        self.initial_vm = initial_vm
        self.print_evaluation_details = print_evaluation_details

        self.channel_output_profile = ChannelOutputProfile(keep_data=False, save_plots=False)
        self.compartment_output_profile = CompartmentOutputProfile(keep_data=True, save_plots=True)

        self.evaluated = False
        self.constraint_pass = (not use_constraints) or self.test_constraints()
        self.score = 0


    def get_active_parameters(self):
        if self.model_type == ModelType.full_model:
            return self.variable_parameters.copy()
        if self.model_type == ModelType.soma_conductance:
            active_parameters = p1.copy()
            active_parameters[50] = self.variable_parameters[0]
            active_parameters[51] = self.variable_parameters[1]
            active_parameters[52] = self.variable_parameters[2]
            active_parameters[53] = self.variable_parameters[3]
            active_parameters[54] = self.variable_parameters[4]
            active_parameters[56] = self.variable_parameters[5]
            return active_parameters
        if self.model_type == ModelType.soma_kna_conductance:
            active_parameters = p1.copy()
            active_parameters[50] = self.variable_parameters[0]
            active_parameters[51] = self.variable_parameters[1]
            return active_parameters

    def initialize_model(self, delta_t):
        # Constants
        cdef double d_leakage_max_g = 0.126
        cdef double s_leakage_max_g = 0.042
        cdef double i_leakage_max_g = 0.0042

        cdef double d_capacitance = 0.942
        cdef double s_capacitance = 0.314
        cdef double i_capacitance = 0.0314

        # Ratios
        cdef double hva_to_lva_calcium_ratio = self.active_parameters[56]
        cdef double s_to_i_fast_potassium_ratio = self.active_parameters[57]
        cdef double s_to_i_fast_sodium_ratio = self.active_parameters[58]
        cdef double d_to_i_coupling_ratio = self.active_parameters[59]

        # Assemble dendrites
        self.d_leakage = LeakageChannel(max_conductance=d_leakage_max_g, channel_preset=ChannelPresets.leakage)
        self.d_compartment = Compartment(compartment_label="Dendrites", capacitance=d_capacitance, initial_vm=self.initial_vm,
                                    channels=[self.d_leakage], output_profile=self.compartment_output_profile)

        # Assemble soma
        self.s_leakage = LeakageChannel(max_conductance=s_leakage_max_g, channel_preset=ChannelPresets.leakage)
        self.s_fast_potassium = VoltageGatedChannel(max_conductance=self.active_parameters[50],
                                               channel_preset=ChannelPresets.fast_potassium,
                                               activation_parameters=self.active_parameters[0:6],
                                               output_profile=self.channel_output_profile)
        self.s_fast_sodium = VoltageGatedChannel(max_conductance=self.active_parameters[51], channel_preset=ChannelPresets.fast_sodium,
                                            activation_parameters=self.active_parameters[6:12],
                                            inactivation_parameters=self.active_parameters[12:18],
                                            output_profile=self.channel_output_profile)
        self.s_slow_sodium = VoltageGatedChannel(max_conductance=self.active_parameters[52], channel_preset=ChannelPresets.slow_sodium,
                                            activation_parameters=self.active_parameters[18:24],
                                            inactivation_parameters=self.active_parameters[24:30],
                                            output_profile=self.channel_output_profile)
        self.s_hva_calcium = VoltageGatedChannel(max_conductance=self.active_parameters[53], channel_preset=ChannelPresets.hva_calcium,
                                            activation_parameters=self.active_parameters[30:36],
                                            output_profile=self.channel_output_profile)
        self.s_lva_calcium = VoltageGatedChannel(max_conductance=(self.active_parameters[53] * hva_to_lva_calcium_ratio),
                                            channel_preset=ChannelPresets.lva_calcium,
                                            activation_parameters=self.active_parameters[36:42],
                                            inactivation_parameters=self.active_parameters[42:48],
                                            output_profile=self.channel_output_profile)
        self.s_calcium_activated_potassium = IonGatedChannel(max_conductance=self.active_parameters[54], decay=self.active_parameters[49],
                                                        channel_preset=ChannelPresets.calcium_activated_potassium,
                                                        activator_channels=[
                                                            (self.s_hva_calcium, self.active_parameters[48]),
                                                            (self.s_lva_calcium, self.active_parameters[48] * hva_to_lva_calcium_ratio)
                                                        ], output_profile=self.channel_output_profile)
        self.s_compartment = Compartment(compartment_label="Soma", capacitance=s_capacitance, initial_vm=self.initial_vm,
                                    channels=[
                                        self.s_leakage, self.s_fast_potassium, self.s_fast_sodium, self.s_slow_sodium,
                                        self.s_hva_calcium, self.s_lva_calcium, self.s_calcium_activated_potassium
                                    ], output_profile=self.compartment_output_profile, stimulus_compartment=True)

        # Assemble initial segment
        self.i_leakage = LeakageChannel(max_conductance=i_leakage_max_g, channel_preset=ChannelPresets.leakage)
        self.i_fast_potassium = VoltageGatedChannel(max_conductance=(self.active_parameters[50] * s_to_i_fast_potassium_ratio),
                                               channel_preset=ChannelPresets.fast_potassium,
                                               activation_parameters=self.active_parameters[0:6],
                                               output_profile=self.channel_output_profile)
        self.i_fast_sodium = VoltageGatedChannel(max_conductance=(self.active_parameters[51] * s_to_i_fast_sodium_ratio),
                                            channel_preset=ChannelPresets.fast_sodium,
                                            activation_parameters=self.active_parameters[6:12],
                                            inactivation_parameters=self.active_parameters[12:18],
                                            output_profile=self.channel_output_profile)
        self.i_compartment = Compartment(compartment_label="Initial Segment", capacitance=i_capacitance,
                                    initial_vm=self.initial_vm,
                                    channels=[self.i_leakage, self.i_fast_potassium, self.i_fast_sodium],
                                    output_profile=self.compartment_output_profile)

        # Assemble model
        self.model = Model(delta_t=delta_t, compartments=[self.d_compartment, self.s_compartment, self.i_compartment],
                      coupling_conductances=[self.active_parameters[55], self.active_parameters[55] * d_to_i_coupling_ratio])


    def evaluate_single_ap_data(self, delta_t, stimulus_start, score_breakdown=False):
        def find_error(value, mean, sd):
            if value / mean < 0:
                return False, 0

            cdef double z = abs((value - mean) / sd)
            if z > allowed_sds and not math.isclose(z, allowed_sds):
                return False, 0

            cdef double s = -1 * np.tanh(z - allowed_sds)
            return True, max(-0.5, s)

        error_arr = []
        test_success = True

        # Vrest Tests
        cdef double vrest_vm = np.mean(self.s_compartment.vm_history[:int(stimulus_start / delta_t)])
        success, vrest_vm_error = find_error(vrest_vm, _vrest_vm, _vrest_vm_sd)
        if self.print_evaluation_details:
            print("Vrest: {} mV".format(vrest_vm))
            print("\t{}".format(vrest_vm_error))
        test_success = test_success and success
        if not test_success:
            return False, 0
        error_arr.append(vrest_vm_error)

        # AP Tests
        cdef int ap_index = np.argmax(self.s_compartment.vm_history[int(stimulus_start / delta_t):]) \
                   + int(stimulus_start / delta_t)
        cdef double ap_vm = self.s_compartment.vm_history[ap_index]
        success, ap_vm_error = find_error(ap_vm - vrest_vm, _ap_vm, _ap_vm_sd)
        if self.print_evaluation_details:
            print("AP: {} mV".format(ap_vm - vrest_vm))
            print("\t{}".format(ap_vm_error))
        test_success = test_success and success
        if not test_success:
            return False, 0
        error_arr.append(ap_vm_error)

        # fAHP Tests
        fAHP_index = find_peaks(np.negative(self.s_compartment.vm_history[ap_index:]),
                                   height=(-1 * (_fAHP_vm + vrest_vm) - detection_sds * _fAHP_vm_sd,
                                           -1 * (_fAHP_vm + vrest_vm) + detection_sds * _fAHP_vm_sd)
                                   )[0]
        if len(fAHP_index) == 0:
            return ap_vm_error > 0, 0
        fAHP_index = fAHP_index[0] + ap_index

        cdef double fAHP_vm = self.s_compartment.vm_history[fAHP_index]
        success, fAHP_vm_error = find_error(fAHP_vm - vrest_vm, _fAHP_vm, _fAHP_vm_sd)
        test_success = test_success and success
        error_arr.append(fAHP_vm_error)

        cdef double fAHP_delay = delta_t * (fAHP_index - ap_index)
        success, fAHP_delay_error = find_error(fAHP_delay, _fAHP_delay, _fAHP_delay_sd)
        test_success = test_success and success
        error_arr.append(fAHP_delay_error)

        if self.print_evaluation_details:
            print("fAHP {} mV ({} ms)".format(fAHP_vm - vrest_vm, fAHP_delay))
            print("\t{} ({})".format(fAHP_vm_error, fAHP_delay_error))

        # ADP Tests
        ADP_index = find_peaks(self.s_compartment.vm_history[fAHP_index:],
                                height=((_ADP_vm + vrest_vm) - detection_sds * _ADP_vm_sd,
                                        (_ADP_vm + vrest_vm) + detection_sds * _ADP_vm_sd)
                                )[0]
        if len(ADP_index) == 0:
            return ap_vm_error > 0, 0
        ADP_index = ADP_index[0] + fAHP_index

        cdef double ADP_vm = self.s_compartment.vm_history[ADP_index]
        success, ADP_vm_error = find_error(ADP_vm - vrest_vm, _ADP_vm, _ADP_vm_sd)
        test_success = test_success and success
        error_arr.append(ADP_vm_error)

        cdef double ADP_delay = delta_t * (ADP_index - ap_index)
        success, ADP_delay_error = find_error(ADP_delay, _ADP_delay, _ADP_delay_sd)
        test_success = test_success and success
        error_arr.append(ADP_delay_error)

        if self.print_evaluation_details:
            print("ADP: {} mV ({} ms)".format(ADP_vm - vrest_vm, ADP_delay))
            print("\t{} ({})".format(ADP_vm_error, ADP_delay_error))

        # fAHP Tests
        # t = np.negative(self.s_compartment.vm_history[ADP_index:])
        # res = []
        # prev = 100
        # for i, val in enumerate(t):
        #     if prev < val:
        #         res.append((i - 1, prev))
        #
        #     prev = val

        sAHP_index = find_peaks(np.negative(self.s_compartment.vm_history[ADP_index:]),
                                height=(-1 * (_sAHP_vm + vrest_vm) - detection_sds * _sAHP_vm_sd,
                                        -1 * (_sAHP_vm + vrest_vm) + detection_sds * _sAHP_vm_sd)
                                )[0]
        if len(sAHP_index) == 0:
            return ap_vm_error > 0, 0
        sAHP_index = sAHP_index[0] + ADP_index

        cdef double sAHP_vm = self.s_compartment.vm_history[sAHP_index]
        success, sAHP_vm_error = find_error(sAHP_vm - vrest_vm, _sAHP_vm, _sAHP_vm_sd)
        test_success = test_success and success
        error_arr.append(sAHP_vm_error)

        cdef double sAHP_delay = delta_t * (sAHP_index - ADP_index)
        success, sAHP_delay_error = find_error(sAHP_delay, _sAHP_delay, _sAHP_delay_sd)
        test_success = test_success and success
        error_arr.append(sAHP_delay_error)

        if self.print_evaluation_details:
            print("sAHP {} mV ({} ms)".format(sAHP_vm - vrest_vm, sAHP_delay))
            print("\t{} ({})".format(sAHP_vm_error, sAHP_delay_error))

        # dv/dt Tests
        dvdt = np.subtract(self.s_compartment.vm_history[1:], self.s_compartment.vm_history[:-1]) / delta_t

        cdef double dvdt_rise = max(dvdt)
        success, dvdt_rise_error = find_error(dvdt_rise, _dvdt_rise, _dvdt_rise_sd)
        test_success = test_success and success
        error_arr.append(dvdt_rise_error)

        cdef double dvdt_fall = min(dvdt)
        success, dvdt_fall_error = find_error(dvdt_fall, _dvdt_fall, _dvdt_fall_sd)
        test_success = test_success and success
        error_arr.append(dvdt_fall_error)

        if self.print_evaluation_details:
            print("dv/dt rise: {}".format(dvdt_rise))
            print("\t{}".format(dvdt_rise_error))
            print("dv/dt fall: {}".format(dvdt_fall))
            print("\t{}".format(dvdt_fall_error))


        # AP half-amplitude duration
        cdef double half_amp_vm = (ap_vm - vrest_vm) / 2 + vrest_vm

        if self.print_evaluation_details:
            print("half-amp vm: {}".format(half_amp_vm))

        cdef double half_amp_1_time = 0
        cdef double half_amp_2_time = 0
        cdef double half_amp_1_time_lin = 0
        cdef double half_amp_2_time_lin = 0

        for i in range(1, len(self.s_compartment.vm_history)):
            if self.s_compartment.vm_history[i - 1] < half_amp_vm <= self.s_compartment.vm_history[i]:
                if half_amp_1_time < 0.001:
                    half_amp_1_time = delta_t * (i - 1)
                    half_amp_1_time_lin = delta_t * (i - 1) + (half_amp_vm - self.s_compartment.vm_history[i - 1]) / ((self.s_compartment.vm_history[i] - self.s_compartment.vm_history[i - 1]) / delta_t)
                    # print("ha1: {} ({})".format(half_amp_1_time, half_amp_1_time_lin))
                    print("ha1: {} ({} = {} + {})".format(half_amp_1_time, half_amp_1_time_lin,
                                                          delta_t * (i - 1), (half_amp_vm - self.s_compartment.vm_history[i - 1]) / ((self.s_compartment.vm_history[i] - self.s_compartment.vm_history[i - 1]) / delta_t)))
            elif self.s_compartment.vm_history[i] < half_amp_vm <= self.s_compartment.vm_history[i - 1]:
                if i > ap_index:
                    half_amp_2_time = delta_t * i
                    half_amp_2_time_lin = delta_t * (i - 1) + (half_amp_vm - self.s_compartment.vm_history[i - 1]) / ((self.s_compartment.vm_history[i] - self.s_compartment.vm_history[i - 1]) / delta_t)
                    # print("ha2: {} ({})".format(half_amp_2_time, half_amp_2_time_lin))
                    print("ha2: {} ({} = {} + {})".format(half_amp_2_time, half_amp_2_time_lin,
                                                          delta_t * (i - 1), (half_amp_vm - self.s_compartment.vm_history[i - 1]) / ((self.s_compartment.vm_history[i] - self.s_compartment.vm_history[i - 1]) / delta_t)))
                    break
        cdef double half_amp_duration = round(half_amp_2_time - half_amp_1_time, 3)
        cdef double half_amp_duration_lin = round(half_amp_2_time_lin - half_amp_1_time_lin, 3)

        if self.print_evaluation_details:
            print("half-amp duration: {} ({})".format(half_amp_duration, half_amp_duration_lin))


        # Return score
        # if not test_success:
        #     return False, 0
        # return True, np.mean(error_arr)
        if score_breakdown:
            return test_success or (ap_vm_error > 0), np.mean(error_arr) if test_success else 0, np.array([
                vrest_vm, ap_vm - vrest_vm,
                fAHP_vm - vrest_vm, fAHP_delay,
                ADP_vm - vrest_vm, ADP_delay,
                sAHP_vm - vrest_vm, sAHP_delay,
                dvdt_rise, dvdt_fall
            ])

        return test_success or (ap_vm_error > 0), np.mean(error_arr) if test_success else 0

    def evaluate_repetitive_firing_data(self, delta_t, stimulus_start, score_breakdown=False):
        # Establish Vrest
        cdef double vrest_vm = np.mean(self.s_compartment.vm_history[:int(stimulus_start / delta_t)])

        # Collect AP gaps
        cdef double initial_freq
        cdef double steady_freq
        cdef double sfa
        ap_indices = find_peaks(self.s_compartment.vm_history,
                               height=((_ap_vm + vrest_vm) - detection_sds * _ap_vm_sd,
                                       (_ap_vm + vrest_vm) + detection_sds * _ap_vm_sd)
                               )[0]
        if len(ap_indices) < 6:
            if self.print_evaluation_details:
                print("Repetitive firing failed: low firing")
            outcome = RepetitiveFiringOutcome.low_firing
        else:
            ap_gaps = np.subtract(ap_indices[1:], ap_indices[:-1]) * delta_t

            # Test for erratic firing
            for i in range(1, len(ap_gaps)):
                if ap_gaps[i] * 1.2 < ap_gaps[i - 1]:
                    if self.print_evaluation_details:
                        print("Repetitive firing failed: erratic firing")
                    if score_breakdown:
                        return RepetitiveFiringOutcome.erratic_firing, np.array([])
                    return RepetitiveFiringOutcome.erratic_firing

            outcome = RepetitiveFiringOutcome.success
            initial_freq = 1000 / ap_gaps[0]
            steady_freq = 1000 / ap_gaps[int(np.ceil(len(ap_gaps) / 2))]
            sfa = (initial_freq - steady_freq) / initial_freq

            if initial_freq < _initial_freq_min:
                if self.print_evaluation_details:
                    print("Repetitive firing failed: low initial firing frequency")
                outcome = RepetitiveFiringOutcome.low_initial_freq
            elif initial_freq > _initial_freq_max:
                if self.print_evaluation_details:
                    print("Repetitive firing failed: high initial firing frequency")
                outcome = RepetitiveFiringOutcome.high_initial_freq
            elif not _sfa_min < sfa < _sfa_max:
                if self.print_evaluation_details:
                    print("Repetitive firing failed: high spike frequency adaptation")
                outcome = RepetitiveFiringOutcome.high_sfa

            if self.print_evaluation_details:
                print("Initial Frequency: {}".format(initial_freq))
                print("Steady Frequency: {}".format(steady_freq))
                print("Spike Frequency Adaptation: {}".format(sfa))


                np.savetxt(os.path.join(self.directory + "_rep", "firing_AP_freqs.csv"),
                           np.array(list(zip(ap_indices[1:] * delta_t, 1000 / ap_gaps))), delimiter=',')
                # print("Spike frequencies:")
                # for t, f in zip(ap_indices[1:], 1000 / ap_gaps):
                #     print("\n{} ms: {} Hz".format(t, f))

        if score_breakdown:
            return outcome, np.array([
                initial_freq, steady_freq, sfa
            ])

        return outcome

    def evaluate_model(self, score_breakdown=False):
        if self.evaluated and not self.print_evaluation_details:
            return self.score

        if not self.constraint_pass:
            self.evaluated = True
            self.score = 0

            return self.score

        directory = str(datetime.datetime.now()).replace(" ", "_")
        self.directory = directory
        if self.print_evaluation_details and \
                (self.channel_output_profile.save_plots or self.compartment_output_profile.save_plots):
            os.system("mkdir {}_ap".format(directory))
            os.system("mkdir {}_rep".format(directory))

        # Evaluate single AP at lowest current
        s_success = False
        cdef double s_score = 0
        s_outcome_dict = {}
        cdef int min_s_stimulus_duration_i = 0
        cdef int max_s_stimulus_duration_i = int(np.ceil(_s_stimulus_duration_max / _s_stimulus_duration_iterator) -
                                         np.floor(_s_stimulus_duration_min / _s_stimulus_duration_iterator))

        cdef double s_stimulus_duration = 0
        while max_s_stimulus_duration_i - min_s_stimulus_duration_i > 0:
            s_stimulus_duration = _s_stimulus_duration_min + _s_stimulus_duration_iterator *\
                                  int((max_s_stimulus_duration_i - min_s_stimulus_duration_i) / 2 +
                                      min_s_stimulus_duration_i)

            if self.print_evaluation_details:
                print("\nSingle AP stimulus duration: {} ms\n".format(s_stimulus_duration))

            if s_stimulus_duration in s_outcome_dict:
                s_success, s_score = s_outcome_dict[s_stimulus_duration]
            else:
                self.initialize_model(delta_t=_s_delta_t)
                self.model.run_simulation(duration=_s_duration, stimulus_current=_s_stimulus,
                                          stimulus_start=_s_stimulus_start, stimulus_duration=s_stimulus_duration)
                s_success, s_score = self.evaluate_single_ap_data(delta_t=_s_delta_t, stimulus_start=_s_stimulus_start)
                s_outcome_dict[s_stimulus_duration] = (s_success, s_score)

            if self.print_evaluation_details:
                print("\nSuccess: {}\nScore: {}\n".format(s_success, s_score))
                print("-" * 50)

            if not s_success:
                min_s_stimulus_duration_i += int((max_s_stimulus_duration_i - min_s_stimulus_duration_i) / 2) + 1
            else:
                if max_s_stimulus_duration_i - min_s_stimulus_duration_i == 1:
                    break

                max_s_stimulus_duration_i -= int((max_s_stimulus_duration_i - min_s_stimulus_duration_i) / 2)

        s_stimulus_duration = _s_stimulus_duration_min + _s_stimulus_duration_iterator * \
                              int((max_s_stimulus_duration_i - min_s_stimulus_duration_i) / 2 +
                                  min_s_stimulus_duration_i)

        if self.print_evaluation_details:
            print("\nSingle AP stimulus duration: {} ms\n".format(s_stimulus_duration))

        self.initialize_model(_s_delta_t)
        self.model.run_simulation(duration=_s_duration, stimulus_current=_s_stimulus,
                                  stimulus_start=_s_stimulus_start, stimulus_duration=s_stimulus_duration)
        if score_breakdown:
            s_success, s_score, s_breakdown = self.evaluate_single_ap_data(delta_t=_s_delta_t, stimulus_start=_s_stimulus_start, score_breakdown=True)
        else:
            s_success, s_score = self.evaluate_single_ap_data(delta_t=_s_delta_t, stimulus_start=_s_stimulus_start)
        if self.print_evaluation_details:
            print("\nSuccess: {}\nScore: {}\n".format(s_success, s_score))
            print("-" * 50)
            self.model.cleanup(directory=(directory + "_ap"))

        # Detect realistic repetitive firing behavior
        r_outcome = None
        cdef int min_r_stimulus_i = 0
        cdef int max_r_stimulus_i = int(np.ceil(_r_stimulus_max / _r_stimulus_iterator) -
                               np.floor(_r_stimulus_min / _r_stimulus_iterator))

        cdef double r_stimulus = 0
        while max_r_stimulus_i - min_r_stimulus_i > 0:
            r_stimulus = _r_stimulus_min + _r_stimulus_iterator * \
                         int((max_r_stimulus_i - min_r_stimulus_i) / 2 + min_r_stimulus_i)

            if self.print_evaluation_details:
                print("\nRepetitive firing stimulus: {} nA\n".format(r_stimulus))

            self.initialize_model(delta_t=_r_delta_t)
            self.model.run_simulation(duration=_r_duration, stimulus_current=r_stimulus,
                                      stimulus_start=_r_stimulus_start, stimulus_duration=_r_stimulus_duration)
            if score_breakdown:
                r_outcome, r_breakdown = self.evaluate_repetitive_firing_data(delta_t=_r_delta_t, stimulus_start=_r_stimulus_start, score_breakdown=True)
            else:
                r_outcome = self.evaluate_repetitive_firing_data(delta_t=_r_delta_t, stimulus_start=_r_stimulus_start)

            if self.print_evaluation_details:
                print("\nOutcome: {}\n".format(r_outcome))
                print("-" * 50)

            if r_outcome in [RepetitiveFiringOutcome.low_firing,
                             RepetitiveFiringOutcome.low_initial_freq,
                             RepetitiveFiringOutcome.high_sfa]:
                min_r_stimulus_i += int((max_r_stimulus_i - min_r_stimulus_i) / 2) + 1
            elif r_outcome is RepetitiveFiringOutcome.high_initial_freq:
                if max_r_stimulus_i - min_r_stimulus_i == 1:
                    break

                max_r_stimulus_i -= int((max_r_stimulus_i - min_r_stimulus_i) / 2)
            elif r_outcome in [RepetitiveFiringOutcome.erratic_firing,
                               RepetitiveFiringOutcome.success]:
                break

        if self.print_evaluation_details:
            self.model.cleanup(directory=(directory + "_rep"))

        self.evaluated = True
        self.score = s_score if s_success and (r_outcome is RepetitiveFiringOutcome.success) else 0

        if score_breakdown:
            return self.score, np.concatenate((s_breakdown, r_breakdown))
        return self.score

    def test_constraints(self):
        self.initialize_model(delta_t=0)

        constraints = [
            # relative_maximum_conductance
            ParameterConstraint(constraint_type=ConstraintType.relative_maximum_conductance,
                                channel_one=self.s_fast_sodium, channel_two=self.s_fast_potassium),
            ParameterConstraint(constraint_type=ConstraintType.relative_maximum_conductance,
                                channel_one=self.s_fast_sodium, channel_two=self.s_slow_sodium),
            # channel_b_parameter_range
            ParameterConstraint(constraint_type=ConstraintType.b_parameter_range,
                                channel_one=self.s_fast_potassium, v_rest=self.initial_vm),
            ParameterConstraint(constraint_type=ConstraintType.b_parameter_range,
                                channel_one=self.s_fast_sodium, v_rest=self.initial_vm),
            ParameterConstraint(constraint_type=ConstraintType.b_parameter_range,
                                channel_one=self.s_slow_sodium, v_rest=self.initial_vm),
            ParameterConstraint(constraint_type=ConstraintType.b_parameter_range,
                                channel_one=self.s_hva_calcium, v_rest=self.initial_vm),
            ParameterConstraint(constraint_type=ConstraintType.b_parameter_range,
                                channel_one=self.s_lva_calcium, v_rest=self.initial_vm),
            # channel_activation_voltage_range, channel_inactivation_voltage_range
            ParameterConstraint(constraint_type=ConstraintType.activation_voltage_range,
                                channel_one=self.s_fast_potassium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.9),
            ParameterConstraint(constraint_type=ConstraintType.activation_voltage_range,
                                channel_one=self.s_fast_sodium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.9),
            ParameterConstraint(constraint_type=ConstraintType.inactivation_voltage_range,
                                channel_one=self.s_fast_sodium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.8),
            ParameterConstraint(constraint_type=ConstraintType.activation_voltage_range,
                                channel_one=self.s_slow_sodium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.9),
            ParameterConstraint(constraint_type=ConstraintType.inactivation_voltage_range,
                                channel_one=self.s_slow_sodium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.8),
            ParameterConstraint(constraint_type=ConstraintType.activation_voltage_range,
                                channel_one=self.s_hva_calcium, v_rest=self.initial_vm,
                                activation_range=(-45, -30), rest_probability=0.9),
            ParameterConstraint(constraint_type=ConstraintType.activation_voltage_range,
                                channel_one=self.s_lva_calcium, v_rest=self.initial_vm,
                                activation_range=(-65, -50), rest_probability=0.9),
            ParameterConstraint(constraint_type=ConstraintType.inactivation_voltage_range,
                                channel_one=self.s_lva_calcium, v_rest=self.initial_vm,
                                activation_range=(-80, -60), rest_probability=0.8),
            # activation_inactivation_intersection
            ParameterConstraint(constraint_type=ConstraintType.activation_inactivation_intersection,
                                channel_one=self.s_fast_sodium),
            ParameterConstraint(constraint_type=ConstraintType.activation_inactivation_intersection,
                                channel_one=self.s_slow_sodium),
            ParameterConstraint(constraint_type=ConstraintType.activation_inactivation_intersection,
                                channel_one=self.s_lva_calcium),
            # inactivation_time_constants
            ParameterConstraint(constraint_type=ConstraintType.internal_time_constants,
                                channel_one=self.s_fast_sodium),
            ParameterConstraint(constraint_type=ConstraintType.internal_time_constants,
                                channel_one=self.s_slow_sodium),
            ParameterConstraint(constraint_type=ConstraintType.internal_time_constants,
                                channel_one=self.s_lva_calcium),
            # relative_activation_time_constants, relative_inactivation_time_constants
            ParameterConstraint(constraint_type=ConstraintType.relative_activation_time_constants,
                                channel_one=self.s_fast_sodium, channel_two=self.s_slow_sodium),
            ParameterConstraint(constraint_type=ConstraintType.relative_inactivation_time_constants,
                                channel_one=self.s_fast_sodium, channel_two=self.s_slow_sodium),
            ParameterConstraint(constraint_type=ConstraintType.relative_activation_time_constants,
                                channel_one=self.s_fast_sodium, channel_two=self.s_fast_potassium),
            ParameterConstraint(constraint_type=ConstraintType.relative_activation_time_constants,
                                channel_one=self.s_lva_calcium, channel_two=self.s_hva_calcium)
        ]

        constraint_pass = True
        for parameter_constraint in constraints:
            constraint_pass = constraint_pass and\
                              parameter_constraint.evaluate_constraint(print_evaluation_details=self.print_evaluation_details)
        if self.print_evaluation_details:
            print("\nConstraints Passed: {}".format(constraint_pass))

        return constraint_pass