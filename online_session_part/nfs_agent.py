import os
import glob
import time
import torch
import pickle
import dill

from torchdump.utils import get_logger

from .agent import OnlineAgent
from .utils import ApiInfo

logger = get_logger()

class NFSAgent(OnlineAgent):
    '''A method of Online Session Part.
    In NAS communication scenario, use shared storage to transport PyTorch Tensors.
    '''
    def __init__(self, nfs_dir=None, load_map_location="cpu") -> None:
        self.nfs_dir = nfs_dir
        self.connection_nums = 0
        self.load_map_location = load_map_location

    def send(self, data):
        assert isinstance(data, (ApiInfo, str)), f"Unexpected data type: {type(data)} to send under online NFS mode. "

        filename = 'rank' + str(data.rank) + '.' + data.name + '.pt' if isinstance(data, ApiInfo) \
                else data + str(time.time_ns())
        outpath = os.path.join(self.nfs_dir, filename)
        # save pt to tmp file first, then replace the original file, this is to avoid the case that the file is not saved completely.
        outpath_tmp = outpath + ".torchdump.tmp"
        try:
            torch.save(data, outpath_tmp, pickle_module=dill,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(outpath_tmp, outpath)
        except Exception as e:
            logger.error(f"Error occurs during saving data to {outpath}, skip it: {e}")
            for p in (outpath_tmp, outpath):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception as e2:
                    logger.error(f"Error occurs during removing {p}: {e2}")

    def _try_load_data(self, glob_type):
        data = None
        file = None
        # Don't use cache - always re-glob to see all current files
        for attempt in range(2):  # Try twice in case file was just being renamed
            files = glob.glob(os.path.join(self.nfs_dir, glob_type))
            if not files:
                return data
            # filter out temporary files (*.torchdump.tmp)
            files = [f for f in files if not f.endswith(".torchdump.tmp")]
            if not files:
                return data
            # find earliest last modified file
            files.sort(key=os.path.getmtime, reverse=True)
            file = files.pop()

            # Check if file still exists before trying to load
            if not os.path.exists(file):
                time.sleep(0.05)
                continue

            try:
                data = torch.load(
                    file, map_location=self.load_map_location, pickle_module=dill)
                break  # Success, no need to retry
            except FileNotFoundError:
                # File was probably just renamed, retry
                time.sleep(0.05)
                continue
            except:
                try:
                    # retry to load in case not saved completely
                    time.sleep(0.1)
                    if not os.path.exists(file):
                        continue
                    if os.path.getsize(file) == 0:
                        logger.warning(f"{file} is empty, skip it.")
                    else:
                        data = torch.load(
                            file, map_location=self.load_map_location, pickle_module=dill)
                    break  # Success or empty file, don't retry
                except FileNotFoundError:
                    # File disappeared, retry
                    time.sleep(0.05)
                    continue
                except Exception as e:
                    logger.error(f"Error occurs during loading data from {file}: {e}")
                    break

        if file is None:
            return data

        # only remove file when loaded successfully or file is empty (corrupted),
        # otherwise keep it for next retry to avoid losing ONLINE_END signal.
        should_remove = False
        if data is not None:
            should_remove = True
        elif os.path.exists(file) and os.path.getsize(file) == 0:
            should_remove = True
        # Do NOT remove other files even if loading fails repeatedly - especially ONLINE_END!
        if should_remove:
            try:
                os.remove(file)
            except Exception as e:
                logger.debug(f"Error occurs during removing {file}: {e}")
        return data

    def recv(self):
        data = self._try_load_data("ONLINE_START*")
        if data == "ONLINE_START":
            self.connection_nums += 1
            # when receiving START, wait for seconds in case reading first empty file
            time.sleep(1)
        if data is not None:
            return data
        # scan .pt file only after received START
        if self.connection_nums > 0:
            for glob_type in ["*.pt", "ONLINE_END*"]:
                data = self._try_load_data(glob_type)
                if data is not None:
                    break
        return data

    def can_stop(self):
        self.connection_nums -= 1
        if self.connection_nums == 0:
            return True
        return False
