
class AdvisorMessageSet:
    @property
    def all_passed(self):
        return "All data in comparison result meets the accuracy requirements."

    @property
    def fwd_input(self):
        return "1. Analyze the model to view the input source.\n" \
               "2. Check if the random seed is fixed, if not, you can use seed_all().\n" \
               "3. The fault may be caused by memory corruption and further analysis is required."

    @property
    def fwd_output(self):
        return "This is a forward API computation error. Check the computation implementation."

    @property
    def bwd_input(self):
        return "Check whether the forward computation result is affected."

    @property
    def bwd_output(self):
        return "This is a backward API computation error. Check the computation implementation."

    @property
    def deterministic(self):
        # To be confirmed
        return ""

    def high_precision(self, env_name):
        return "This API supports running in high-precision mode on MLU devices. " \
               f"Please export {env_name}=1 to verify if the accuracy meets the requirements."

    @property
    def warning(self):
        return '(Please double check the result. ' \
               'It is recommended to set dump_level="HIGH" when dumping data to improve analysis accuracy.)'


AdvisorMessage = AdvisorMessageSet()
