from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice
from ffcv.pipeline.operation import Operation

import torch

from tqdm import tqdm


# custom pipline transforms
# TODO


class PickACorner(Operation):
    # Return the code to run this operation
    @abstractmethod
    def generate_code(self) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        raise NotImplementedError


BATCH_SIZE = 128
NUM_WORKERS = 4
ORDERING = OrderOption.QUASI_RANDOM

device = torch.device("cpu")

PIPELINES = {
    "audio": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
}

loader = Loader(
    "/scratch/users/nshaheed/mtg-jamendo-ffcv.beton",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    order=ORDERING,
    pipelines=PIPELINES,
    os_cache=False,  # can't cache bc it's too big to fit in memory
)


for val in tqdm(loader, smoothing=0):
    pass
