from actoris_harena import Agent
from actoris_harena import TrainableAgent
import numpy as np
from actoris_harena.utilities.visual_utils \
    import draw_pick_and_place
from matplotlib import pyplot as plt

class WorkspaceSampler(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.agent = config.agent
        if isinstance(self.agent, TrainableAgent):  
            self.agent.load_checkpoint(config.checkpoint)
    
    def reset(self, arena_ids):
        self.agent.reset(arena_ids)
    
    def get_state(self):
        return self.agent.get_state()
    
    def init(self, infos):
        pass #self.agent.init(infos)
    
    def update(self, infos, actions):
        pass #self.agent.update(infos, actions)
        #self.agent.update(infos, actions)
    
    def terminate(self):
        return self.agent.terminate()
    
    def success(self):
        return self.agent.success()
    
    def act(self, infos, update=False):
        self.raw_infos = infos
        self.new_infos = []
        for info in infos:
            new_info = self._sample_workspace(info)
            self.new_infos.append(new_info)

        self.agent.init(self.new_infos)
        actions = self.agent.act(self.new_infos)
        ret_actions = []
        for i in range(len(actions)):
            r_act = self._remap_action(actions[i], self.raw_infos[i], self.new_infos[i])
            ret_actions.append(r_act)
        return ret_actions
        

    def _sample_workspace(self, info):
        # Sample workspace
        rgb = info['observation']['rgb']
        depth = info['observation']['depth']
        mask = info['observation']['mask']
        # print(info['observation'].keys())
        workspace_mask = info['observation']['workspace_mask']
        H, W = mask.shape
        new_info = {'observation': {}}
        L = min(H, W)

        ## get the intersection between mask and workspace_mask
        cloth_workspace_mask = np.logical_and(workspace_mask, mask)

        ## find the center of the above mask
        x, y = np.where(cloth_workspace_mask)
        y_center = int(np.mean(y))
        x_center = int(np.mean(x))

        crop_x = max(x_center - L//2, 0)
        crop_y = max(y_center - L//2, 0)
        ## make sure crop_x + L < H and crop_y + L < W
        crop_x = min(crop_x, H - L)
        crop_y = min(crop_y, W - L)

        new_info['observation']['rgb'] = rgb[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['depth'] = depth[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['mask'] = cloth_workspace_mask[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['workspace_mask'] = workspace_mask[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['crop_x'] = crop_x
        new_info['crop_y'] = crop_y
        new_info['crop_length'] = L
        new_info['arena_id'] = info['arena_id']

        
        plt.imsave('tmp/raw_rgb.png', rgb)
        plt.imsave('tmp/crop_rgb.png', new_info['observation']['rgb'])
        plt.imsave('tmp/raw_depth.png', depth.reshape(H, W, 1).repeat(3, axis=2))
        plt.imsave('tmp/crop_depth.png', new_info['observation']['depth'].reshape(L, L, 1).repeat(3, axis=2))
        # max and min depth of cropped depth
        #print(np.max(new_info['observation']['depth']), np.min(new_info['observation']['depth']))
        plt.imsave('tmp/raw_mask.png', mask.reshape(H, W, 1).repeat(3, axis=2).astype(np.float32))
        plt.imsave('tmp/crop_mask.png', new_info['observation']['mask'].reshape(L, L, 1).repeat(3, axis=2).astype(np.float32))
        plt.imsave('tmp/raw_workspace_mask.png', workspace_mask.reshape(H, W, 1).repeat(3, axis=2).astype(np.float32))
        
        return new_info
    
    def get_phase(self):
        return self.agent.get_phase()

    def _remap_action(self, action, raw_info, new_info):
        #print(action)

        crop_x = new_info['crop_x']
        crop_y = new_info['crop_y']
        crop_length = new_info['crop_length']
        H, W = raw_info['observation']['mask'].shape

        # pixel coordinates on the cropped observation
        action = action['norm-pixel-pick-and-place']
        pick_0 = action['pick_0']
        place_0 = action['place_0']
        
        # map the pick and place actions to the original observation
        pick_0 = (pick_0 + 1)/2*crop_length
        place_0 = (place_0 + 1)/2*crop_length

        crop_pnp = draw_pick_and_place(
            new_info['observation']['rgb'], 
            pick_0, 
            place_0,
            get_ready=True)
        
        pick_0 += np.array([crop_x, crop_y])
        place_0 += np.array([crop_x, crop_y])
        # map the place action to the nearest pixel on the workspace mask
        place_0 = self._nearest_pixel(
            place_0, raw_info['observation']['workspace_mask'])

        
        raw_rgb = raw_info['observation']['rgb'].astype(np.float32)/255
        workspace = raw_info['observation']['workspace_mask'].astype(np.float32).reshape(H, W, 1)
        alpha = 0.5
        
        raw_rgb = alpha*raw_rgb + (1-alpha)*workspace
        raw_rgb = (raw_rgb*255).astype(np.uint8)

        raw_pnp = draw_pick_and_place(
            raw_rgb, 
            pick_0, 
            place_0,
            get_ready=True)
        
        plt.imsave('tmp/crop_pnp.png', crop_pnp)
        plt.imsave('tmp/raw_pnp.png', raw_pnp)


        # remap the actions to the original observation in space -1 to 1
        action['pick_0'] = pick_0*2/np.asarray([H, W]) - 1
        action['place_0'] = place_0*2/np.asarray([H, W]) - 1

        return {
            'norm-pixel-pick-and-place': action
        }

    def _nearest_pixel(self, point, mask):
        x, y = int(point[0]), int(point[1])
        if mask[x, y]:
            return np.array([x, y])
    
        # get the x and y coordinates of the nearest pixel on the mask
        x, y = np.where(mask)
        dist = np.abs(x - point[0]) + np.abs(y - point[1])
        nearest_idx = np.argmin(dist)

        return np.array([x[nearest_idx], y[nearest_idx]])