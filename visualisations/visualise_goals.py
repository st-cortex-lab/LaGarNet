import actoris_harena.api as ag_ar
from matplotlib import pyplot as plt
import argparse

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--domain', default='clothfunnels-realadapt', type=str)
    return parser.parse_args()

def main():
    # arg parse the object name
    parser = argument_parser()
    arena_name = f'softgym|domain:{parser.domain}-{parser.object},task:flattening,horizon:20'
    arena = ag_ar.build_arena(arena_name + ',disp:{}'.format(False))
    arena.set_eval()
    info = arena.reset()

    if parser.domain == 'clothfunnels-realadapt':

        goal_rgb = info['goal']['rgb']
        plt.imsave(f'tmp/clothfunnels_goal_{parser.object}.png', goal_rgb)
    else:
        rgb = info['observation']['rgb']
        plt.imsave(f'tmp/cur_rgb_{parser.object}.png', rgb)
        goal_rgb = info['goal']['rgb']
        cloth_mask = info['goal']['mask'].squeeze(-1).astype(float)
        plt.imsave(f'tmp/goal_{parser.object}.png', goal_rgb)
        plt.imsave(f'tmp/goal_{parser.object}_mask.png', cloth_mask)
        #print(info['observation'].keys())
        workmask = info['observation']['workspace_mask'].astype(float)
        plt.imsave(f'tmp/goal_{parser.object}_workmask.png', workmask)


        workspace_cloth_mask = workmask * cloth_mask

        workspace_cloth_mask = workspace_cloth_mask[..., None].repeat(3, axis=2)
        workmask = workmask[..., None].repeat(3, axis=2)
        # make the area in workspace mask and cloth-mask have the original colour, 
        # and highlight the other part of workspace with gray colour
        alpha = 0.5
        rgb_to_save =  workspace_cloth_mask * (1.0*goal_rgb/255) + \
            (workmask - workspace_cloth_mask) * alpha + \
            (1-workmask) * (1-alpha) * goal_rgb/255
        rgb_to_save = (rgb_to_save*255).astype('uint8')

        plt.imsave(f'tmp/goal_{parser.object}_final.png', rgb_to_save)
 

if __name__ == '__main__':
    main()