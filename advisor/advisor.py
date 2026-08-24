import os

from torchdump.utils import get_logger

from .advisor_result import AdvisorResult
from .advisor_message import AdvisorMessage

logger = get_logger()

class Advisor:
    def __init__(self, input_data, out_path="", suffix=""):
        self.input_data = input_data
        self.out_path = os.path.realpath(out_path)
        self.suffix = suffix.replace('/', '_')
    
    @staticmethod
    def non_deterministic_apis():
        # To be confirmed
        return []

    @staticmethod
    def deterministic_advisor(message, node_name):
        for api_name in Advisor.non_deterministic_apis():
            if api_name in node_name:
                return AdvisorMessage.deterministic
        return message

    @staticmethod
    def high_precision_advisor(message, node_name):
        if "fused_adam" in node_name or "FusedAdam" in node_name:
            message = AdvisorMessage.high_precision("TORCH_MLU_APEX_ADAM_HIGH_PRECISION")
        elif "sqrt" in node_name:
            message = AdvisorMessage.high_precision("CNNL_ACC_SQRT")
        return message

    def analyze_unmatched(self, analyze_data):
        accuracy_unmatched = analyze_data[analyze_data["Result"] == "UNMATCHED"]
        num_unmatch = len(accuracy_unmatched)
        if num_unmatch != 0:
            for i in range(len(accuracy_unmatched)):
                item = accuracy_unmatched.iloc[i]
                logger.warning("The tensor name matches but the shape or dtype " \
                              "does not match: {}".format(item["Dev0 Name"]))

    def analyze_nan(self, analyze_data):
        accuracy_nan = analyze_data[analyze_data["Result"] == "NAN"]
        num_nan = len(accuracy_nan)
        if num_nan != 0:
            for i in range(len(accuracy_nan)):
                item = accuracy_nan.iloc[i]
                logger.warning("Cannot compare the tensor: {}, the data has NaN.".format(item["Dev0 Name"]))

    def gen_advisor_message(self, node_name):
        if "fwd" in node_name:
            if "input" in node_name:
                message = AdvisorMessage.fwd_input
            else:
                message = AdvisorMessage.fwd_output
                message = self.deterministic_advisor(message, node_name)
                message = self.high_precision_advisor(message, node_name)
        else:
            if "input" in node_name:
                message = AdvisorMessage.bwd_input
            else:
                message = AdvisorMessage.bwd_output
                message = self.deterministic_advisor(message, node_name)
                message = self.high_precision_advisor(message, node_name)
        return message

    def append_warning(self, message):
        return message + "\n" + AdvisorMessage.warning

    def gen_advisor_result(self, pd_data, warning):
        first_failing_data = pd_data.iloc[0]
        node_name = first_failing_data["Dev0 Name"]
        index = first_failing_data['index']
        message = self.gen_advisor_message(node_name)
        logger.warning("Find %s accuracy not reached, the index is %s" % (node_name, index))
        if warning:
            message = self.append_warning(message)
        result = AdvisorResult(node_name, index, message)
        return result

    def analysis(self):
        analyze_data = self.input_data.reset_index()
        self.analyze_unmatched(analyze_data)
        self.analyze_nan(analyze_data)
        failing_data = analyze_data[analyze_data["Result"] == "FAILED"]
        warning_data = analyze_data[analyze_data["Result"] == "WARNING"]
        if failing_data.empty and warning_data.empty:
            result = AdvisorResult("NA", "NA", AdvisorMessage.all_passed)
        elif not failing_data.empty:
            result = self.gen_advisor_result(failing_data, warning=False)
        elif not warning_data.empty:
            result = self.gen_advisor_result(warning_data, warning=True)

        message_list = result.print_advisor_log()
        result.gen_summary_file(self.out_path, message_list, suffix=self.suffix)

