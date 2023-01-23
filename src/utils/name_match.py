from src.learners.baselines.cassle import CASSLELearner
from src.learners.baselines.gdumb import GDUMBLearner
from src.learners.baselines.lump import LUMPLearner
from src.learners.baselines.scr import SCRLearner
from src.learners.baselines.agem import AGEMLearner
from src.learners.baselines.stam.stam import STAMLearner
from src.learners.baselines.er import ERLearner

from src.learners.supcon import SupConLearner
from src.learners.ce import CELearner
from src.learners.simclr import SimCLRLearner
from src.learners.aug import AugLearner

from src.buffers.reservoir import Reservoir
from src.buffers.greedy import GreedySampler
from src.buffers.fifo import QueueMemory


learners = {
    'ER':   ERLearner,
    'CE':   CELearner,
    'SC':   SupConLearner,
    'SCR':  SCRLearner,
    'SimCLR': SimCLRLearner,
    'STAM': STAMLearner,
    'AUG': AugLearner,
    'AGEM': AGEMLearner,
    'LUMP': LUMPLearner,
    'GDUMB': GDUMBLearner,
    'CASSLE': CASSLELearner
}

buffers = {
    'reservoir': Reservoir,
    'greedy': GreedySampler,
    'fifo': QueueMemory,
}
