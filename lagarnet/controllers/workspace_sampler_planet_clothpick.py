import os
import numpy as np
from agent_arena.utilities.visual_utils \
    import draw_pick_and_place
from agent_arena import TrainableAgent

from matplotlib import pyplot as plt

from .workspace_sampler import WorkspaceSampler
from .cloth_mask_workspace_pick_and_place_mpc import ClothMaskWorkspacePickAndPlaceMPC

"""

For each step, sample around 10 workspace and feed them into PlaNetClothPick.

Return the action that produces the highest reward estimation.

"""

class WorkspaceSamplerPlaNetClothPick(WorkspaceSampler):
    def __init__(self, config):
        super().__init__(config)
        self.agent = config.agent
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        #
        planning_config = self.config.policy.params
        planning_config.model = self
        planning_config.action_space = self.config.action_space
        planning_config.cost_fn = 'from_model'
        #planning_config.swap_action = self.config.policy.params.swap_action
        #print(planning_config)
        #logging.info('[rssm, init] action space'.format(self.config.action_space))
        self.agent.planning_algo = ClothMaskWorkspacePickAndPlaceMPC(planning_config)
        self.prioritise_middle = self.config.get('prioritise_middle', False)
        self.continue_internal_updates = self.config.get('continue_internal_updates', False)
        
        #print('continue internal updates!!!!!!!\n!!!!!!!!', self.continue_internal_updates)
        if isinstance(self.agent, TrainableAgent):
            if config.checkpoint == -1:
                self.agent.load()
            else:
                self.agent.load_checkpoint(config.checkpoint)

    def unroll_action_from_cur_state(self, actions, state):
        return self.agent.unroll_action_from_cur_state(actions, state)

    def cost_fn(self, trajectory, goal=None):
        return self.agent.cost_fn(trajectory, goal)
    
    def reset(self, arena_ids):
        self.agent.reset(arena_ids)
        self.step = 0
    
    def get_state(self):
        return self.agent.get_state()
    
    
    def act(self, infos, update=False):
        self.raw_infos = infos
        self.new_infos = []
        ret_actions = []
        
        for info in infos:

            # Generate a list of new infos with different arena_ids and workspace
            new_infos = self._sample_workspace(info)
            #print('new_infos:', len(new_infos))
            arena_ids = [info['arena_id'] for info in new_infos]

            # Always refresh the agent's state
            if not self.continue_internal_updates:
                self.agent.reset(arena_ids)
                self.agent.init(new_infos)
            else:
                if self.step == 0:
                    self.agent.init(new_infos)
                else:
                    self.agent.update(new_infos, self.last_actions)


            actions = self.agent.act(new_infos)
            self.step += 1
            internal_states = self.agent.get_state()
            #print('internal_states:', internal_states)

            if isinstance(self.agent, TrainableAgent):
                est_cost = [internal_states[aid]['action_cost'] for aid in arena_ids]

                # Find the action that produces the highest reward estimation
                best_idx = np.argmin(est_cost)
                best_action = actions[best_idx]
               
            else:
                best_action = actions[0]
                best_idx = 0
            r_act = self._remap_action(best_action, info, new_infos[best_idx])
            ret_actions.append(r_act)
        self.last_actions = ret_actions
       

        return ret_actions
            
        

    def _sample_workspace(self, info):
        # Sample workspace
        rgb = info['observation']['rgb']
        depth = info['observation']['depth']
        mask = info['observation']['mask'].squeeze()
        workspace_mask = info['observation']['workspace_mask']

        H, W = mask.shape
        L = min(H, W)

        cloth_workspace_mask = np.logical_and(workspace_mask, mask)
        assert np.sum(cloth_workspace_mask) > 0, 'No intersection between mask and workspace_mask'

        goal_depth_ = None
        goal_mask_ = None
        goal_rgb_ = None
        if 'goal' in info:
            #print('goal in info')
            goal_depth = info['goal']['depth']
            goal_mask = info['goal']['mask']
            goal_rgb = info['goal']['rgb']
            goal_depth_ = goal_depth[H//2 - L//2:H//2 + L//2, W//2 - L//2:W//2 + L//2]
            goal_mask_ = goal_mask[H//2 - L//2:H//2 + L//2, W//2 - L//2:W//2 + L//2]
            goal_rgb_ = goal_rgb[H//2 - L//2:H//2 + L//2, W//2 - L//2:W//2 + L//2]
        
        new_infos = []
        #print('prioritise_middle:', self.prioritise_middle)
        if self.prioritise_middle:
            y_centre = W//2
            x_centre = H//2
            crop_x = max(x_centre - L//2, 0)
            crop_y = max(y_centre - L//2, 0)
            mask_ = mask[crop_x:crop_x+L, crop_y:crop_y+L]
            if np.sum(mask_) > 0:
                #print('Choose the middle window')
                new_infos.append(
                    self._get_new_info(crop_x, crop_y, L, rgb, depth, mask, 
                                    cloth_workspace_mask, workspace_mask, 
                                    info['arena_id']*100, goal_depth_, goal_mask_, goal_rgb_)
                )
                return new_infos
        
        ## find the center of the above mask
        x, y = np.where(cloth_workspace_mask)
        y_cloth_center = int(np.mean(y))
        x_cloth_center = int(np.mean(x))

        crop_x = max(x_cloth_center - L//2, 0)
        crop_y = max(y_cloth_center - L//2, 0)
        ## make sure crop_x + L < H and crop_y + L < W
        crop_x = min(crop_x, H - L)
        crop_y = min(crop_y, W - L)


        offset_left = -int(self.config.get('offset_left_ratio', 0) * W)
        offset_right = int(self.config.get('offset_right_ratio', 0) * W)
        interval = int(self.config.get('sample_interval_ratio', 0) * W)
        #print('offset_left, offset_right, interval:', offset_left, offset_right, interval)
        if interval == 0:
            offset_ys = [0]
        else:
            offset_ys = list(range(offset_left, offset_right, interval))
        
        if 0 not in offset_ys:
            offset_ys.append(0)
        
        #print('offset_ys:', offset_ys)
        
        for i, ofy in enumerate(offset_ys):

            y_c = y_cloth_center + ofy
            crop_y = max(y_c - L//2, 0)
            crop_y = min(crop_y, W - L)
            
            new_infos.append(
                self._get_new_info(crop_x, crop_y, L, rgb, depth, mask, 
                                   cloth_workspace_mask, workspace_mask, 
                                   info['arena_id']*100 + i, goal_depth_, goal_mask_, goal_rgb_)
            )
        return new_infos
    
    def _get_new_info(self, crop_x, crop_y, L, rgb, depth, mask, 
                      cloth_workspace_mask, workspace_mask, arena_id, 
                      goal_depth_=None, goal_mask_=None, goal_rgb_=None):
        new_info = {'observation': {}}
        new_info['observation']['mask'] = mask[crop_x:crop_x+L, crop_y:crop_y+L] 
        new_info['observation']['cloth-workspace-mask'] = cloth_workspace_mask[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['rgb'] = rgb[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['depth'] = depth[crop_x:crop_x+L, crop_y:crop_y+L]
        new_info['observation']['workspace_mask'] = workspace_mask[crop_x:crop_x+L, crop_y:crop_y+L]
        for key in new_info['observation'].keys():
            #print('key:', key, 'shape:', new_info['observation'][key].shape)
            # Special case for RGB arrays (3 channels)
            arr = new_info['observation'][key]
            pad_width = [(self.config.padding, self.config.padding),
                    (self.config.padding, self.config.padding)]
            
            if arr.ndim == 3:
                pad_width.append((0, 0))

            new_info['observation'][key] = np.pad(
                arr,
                pad_width,
                'constant',
                constant_values=0)
            
            image_to_plot = new_info['observation'][key].copy()
            if key == 'depth':
                # squeeze the depth image to 2D
                image_to_plot = image_to_plot.squeeze()
            # lot save the images
            # create tmp directory if not exists
            plt.imsave(f'tmp/sample_{key}.png', image_to_plot)

        # plt the original image but the cropped image is dimmed
        rgb_ = rgb.copy()/255.0*0.5
        cropped_rgb = rgb[crop_x:crop_x+L, crop_y:crop_y+L].copy()/255.0*0.5
        rgb_[crop_x:crop_x+L, crop_y:crop_y+L] += cropped_rgb
        plt.imsave('tmp/sample_rgb_overlay.png', (rgb_*255).astype(np.uint8))
        new_info['observation']['overlay_rgb'] = (rgb_*255).astype(np.uint8)
        
        

        new_info['crop_x'] = crop_x
        new_info['crop_y'] = crop_y
        new_info['crop_length'] = L
        new_info['padding'] = self.config.padding
        #print('arena_id:', arena_id)
        new_info['arena_id'] = arena_id

        if goal_depth_ is not None:
            new_info['observation']['goal-depth'] = np.pad(
                goal_depth_,
                ((self.config.padding, self.config.padding),
                 (self.config.padding, self.config.padding), 
                 (0, 0)),
                'constant',
                constant_values=0)
            plt.imsave('tmp/sample_goal_depth.png', new_info['observation']['goal-depth'].squeeze())

        if goal_mask_ is not None:
            new_info['observation']['goal-mask'] = np.pad(
                goal_mask_,
                ((self.config.padding, self.config.padding),
                    (self.config.padding, self.config.padding)),
                'constant',
                constant_values=0)
            plt.imsave('tmp/sample_goal_mask.png', new_info['observation']['goal-mask'].squeeze())

        if goal_rgb_ is not None:
            new_info['observation']['goal-rgb'] = np.pad(
                goal_rgb_,
                ((self.config.padding, self.config.padding),
                    (self.config.padding, self.config.padding), (0, 0)),
                'constant',
                constant_values=0)
            plt.imsave('tmp/sample_goal_rgb.png', new_info['observation']['goal-rgb'])
        #print('new info', new_info['observation'].keys())
        return new_info
    
    def get_phase(self):
        return self.agent.get_phase()

    def _remap_action(self, action, raw_info, new_info):
        #print(action)

        crop_x = new_info['crop_x']
        crop_y = new_info['crop_y']
        crop_length = new_info['crop_length']
        padding = new_info['padding']
        length = crop_length + 2*padding
        mask = raw_info['observation']['mask'].squeeze()
        H, W = mask.shape

        # pixel coordinates on the cropped observation
        action = action['norm-pixel-pick-and-place']
        pick_0 = action['pick_0']
        place_0 = action['place_0']
        
        # map the pick and place actions to the original observation
        pick_0 = (pick_0 + 1)/2*length
        place_0 = (place_0 + 1)/2*length

        crop_pnp = draw_pick_and_place(
            new_info['observation']['rgb'], 
            pick_0, 
            place_0,
            get_ready=True)
        
        pick_0 += np.array([crop_x, crop_y])
        place_0 += np.array([crop_x, crop_y])
        pick_0 -= padding
        place_0 -= padding
        # map the place action to the nearest pixel on the workspace mask
        place_0 = self._nearest_pixel(
            place_0, raw_info['observation']['workspace_mask'])

        
        overlay_rgb = new_info['observation']['overlay_rgb'].astype(np.float32)/255
        workspace = raw_info['observation']['workspace_mask'].astype(np.float32).reshape(H, W, 1)
        alpha = 0.5
        
        overlay_rgb = workspace*overlay_rgb + (1 - workspace)*overlay_rgb*alpha
        overlay_rgb = (overlay_rgb*255).astype(np.uint8)

        raw_pnp = draw_pick_and_place(
            overlay_rgb, 
            pick_0, 
            place_0,
            get_ready=True)
        
        plt.imsave('tmp/crop_pnp.png', crop_pnp)
        plt.imsave('tmp/overlay_pnp.png', raw_pnp)


        # remap the actions to the original observation in space -1 to 1
        action['pick_0'] = pick_0*2/np.asarray([H, W]) - 1
        action['place_0'] = place_0*2/np.asarray([H, W]) - 1

        return {
            'norm-pixel-pick-and-place': action
        }

    def _nearest_pixel(self, point, mask):
        x, y = int(point[0]), int(point[1])
        x = min(mask.shape[0]-1, x)
        y = min(mask.shape[1]-1, y)
        if mask[x, y]:
            #print('There is no need to reajust!!!!')
            return np.array([x, y])
    
        # get the x and y coordinates of the nearest pixel on the mask
        x, y = np.where(mask)
        dist = np.abs(x - point[0]) + np.abs(y - point[1])
        nearest_idx = np.argmin(dist)

        return np.array([x[nearest_idx], y[nearest_idx]])