# Codes are borrowed from https://github.com/zxhuang97/medor/tree/main

from agent_arena import TrainableAgent
from .utils import *

class MedorAdaptor(TrainableAgent):

    def __init__(config):
        # TODO
        pass
        self. pick_and_place_num

    def reset(self, arena_ids):
        #TODO
        pass

    def init(self, info_list):
        #TODO
        pass

    def update(info_list, actions):
        #TODO
        pass

    def _act

    def act(info_list, update=False):
        
        data = compute_cloth_mesh_for_planning(plan_cfg, medor_model, matrix_world_to_camera, finetune_cfg, env,
                                                          gt_ds_id, gt_ds_edges, gt_face)
        
        #TODO
        pass

    def load(self, path = None):
        #TODO
        pass

    def set_train(self):
        #TODO
        pass

    def set_eval(self):
        
        #TODO
        pass

    def load_checkpoint(self, checkpoint):
        # This is not needed
        pass

    def save(self, path=None):
        # This is not needed
        pass 

    def get_name(self):
        return "Off-the-shelf Medor"

    def train(config):
        # We only use the off-the-shelf chekcpoint for testing
        pass

    