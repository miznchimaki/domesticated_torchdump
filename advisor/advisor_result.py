import os
import time

from torchdump.utils import get_logger

logger = get_logger()

class AdvisorResult:
    def __init__(self, node, index, message):
        self.suspect_node = node
        self.index = index
        self.advisor_message = message

    @staticmethod
    def gen_summary_file(out_path, message_list, suffix):
        file_name = 'advisor_{}{}.txt'.format(suffix if not suffix or suffix.endswith("_") else suffix + "_",
                                              time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        result_file = os.path.join(out_path, file_name)
        try:
            with open(result_file, 'w+') as output_file:
                output_file.truncate(0)
                message_list = [message + "\n" for message in message_list]
                output_file.writelines(message_list)
            os.chmod(result_file, 0o644)
        except IOError as io_error:
            logger.warning("Failed to save %s, the reason is %s." % (result_file, io_error))
        else:
            logger.info("INFO: The advisor summary is saved in: %s" % result_file)

    def print_advisor_log(self):
        logger.info("INFO: The summary of the expert advice is as follows: ")
        message_list = [
            "Node Index: " + str(self.index),
            "Suspect Nodes: " + self.suspect_node,
            "Expert Advice: " + self.advisor_message
        ]
        for message in message_list:
            logger.info(message)
        return message_list
