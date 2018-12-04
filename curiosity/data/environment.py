'''
Environment object for interaction with 3World
'''

import time
import socket
import numpy as np
import zmq
import copy
import pymongo
from bson.objectid import ObjectId
from PIL import Image
from scipy.misc import imresize
from tdw_client import TDW_Client
import signal
import os
import random
SELECTED_BUILD = ':\Users\mrowca\Desktop\world\one_world.exe'
from collections import namedtuple
#from curiosity.data import flex_environment_utils, generate_grouping, old_assets_query_result_example
import old_assets_query_result_example

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import json
import copy
import cPickle
from pdb import set_trace

synset_for_table = [[u'n04379243']]
rollie_synsets = [[u'n03991062'], [u'n02880940'], [u'n02946921'], [u'n02876657'], [u'n03593526']]
other_vaguely_stackable_synsets = [[u'n03207941'], [u'n04004475'], [u'n02958343'], 
[u'n03001627'], [u'n04256520'], [u'n04330267'], [u'n03593526'], [u'n03761084'], 
[u'n02933112'], [u'n03001627'], [u'n04468005'], [u'n03691459'], [u'n02946921'],
 [u'n03337140'], [u'n02924116'], [u'n02801938'], [u'n02828884'], [u'n03001627'], 
 [u'n04554684'], [u'n02808440'], [u'n04460130'], [u'n02843684'], [u'n03928116']]

shapenet_inquery = {'type': 'shapenetremat', 'has_texture': True, 'version': 0, 'complexity': {'$exists': True}, 'center_pos': {'$exists': True}, 'boundb_pos': {'$exists': True}, 'isLight': {'$exists': True}, 'anchor_type': {'$exists': True}, 'aws_address': {'$exists': True}}
dosch_inquery = {'type': 'dosch', 'has_texture': True, 'version': 1, 'complexity': {'$exists': True}, 'center_pos': {'$exists': True}, 'boundb_pos': {'$exists': True}, 'isLight': {'$exists': True}, 'anchor_type': {'$exists': True}, 'aws_address': {'$exists': True}}




default_keys = ['boundb_pos', 'isLight', 'anchor_type', 'aws_address', 'complexity', 'center_pos']


table_query = copy.deepcopy(shapenet_inquery)
table_query['synset'] = {'$in' : synset_for_table}
rolly_query = copy.deepcopy(shapenet_inquery)
rolly_query['synset'] = {'$in' : rollie_synsets}
other_reasonables_query = copy.deepcopy(shapenet_inquery)
other_reasonables_query['synset'] = {'$in' : other_vaguely_stackable_synsets}

query_dict = {'SHAPENET' : shapenet_inquery, 'ROLLY' : rolly_query, 'TABLE' : table_query, 'OTHER_STACKABLE' : other_reasonables_query}


def query_results_to_unity_data(query_results, scale, mass, var = .01, seed = 0):
        item_list = []
        for i in range(len(query_results)):
                res = query_results[i]
                item = {}
                item['type'] = res['type']
                item['has_texture'] = res['has_texture']
                item['center_pos'] = res['center_pos']
                item['boundb_pos'] = res['boundb_pos']
                item['isLight'] = res['isLight']
                item['anchor_type'] = res['anchor_type']
                #print(res['aws_address'])
                item['aws_address'] = res['aws_address']
                item['mass'] = mass
                item['scale'] = {"option": "Absol_size", "scale": scale, "var": var, "seed": seed, 'apply_to_inst' : True}
                item['_id_str'] = str(res['_id'])
                item_list.append(item)
#        for item in item_list:
#            for k, v in item.items():
#                print(k)
#                print(v)
#        raise Exception('Got what we wanted!')
        return item_list



def init_msg(n_frames):
        msg = {'n': n_frames, 'msg': {"msg_type": "CLIENT_INPUT", "get_obj_data": True, "send_scene_info" : False, "actions": []}}
        msg['msg']['vel'] = [0, 0, 0]
        msg['msg']['ang_vel'] = [0, 0, 0]
        msg['msg']['action_type'] = 'NO_OBJ_ACT'
        return msg


SHADERS = [{'DisplayDepth': 'png'}, {'GetIdentity' : 'png'}, {'Images' : 'png'}]
HDF5_NAMES = [{'DisplayDepth' : 'depths'}, {'GetIdentity' : 'objects'}, {'Images' : 'images'}]

class Environment(object):
    def __init__(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        '''Returns observation after taking action.
        And a boolean representing whether .reset() should
        be called.
        '''
        raise NotImplementedError()


class TDWClientEnvironment(Environment):
    '''
        A barebones interface with tdw server.
        step accepts messages formatted for socket communication.

        unity_seed: random seed to be handed over to unity.
        screen_dims: observation dimensions (height x width)
        room_dims: height by width of the room.
        host_address: address of server for rendering
        selected_build: which binary to use
        msg_names: message name translation
        shaders: shader to data type
        n_cameras: number of cameras in simulation
        gpu_num: gpu on which we render
    '''
    def __init__(self,
            unity_seed,
            random_seed,
            host_address,
            selected_build,
            screen_dims = (128, 170),
            room_dims = (20., 20.),
            msg_names = HDF5_NAMES,
            shaders = SHADERS,
            n_cameras = 1,
            gpu_num = '0',
            environment_timeout = 60,
            image_dir = "",
        ):
        for k_ in [
                'unity_seed',
                'random_seed',
                'host_address',
                'shaders',
                'selected_build',
                'image_dir']:
            setattr(self, k_, eval(k_))

        self.num_steps = 0
        assert isinstance(msg_names, list)
        assert isinstance(shaders, list)
        if self.image_dir:
            if 'Images' not in [s.keys()[0] for s in shaders]:
                shaders.append({'Images': 'png'})
            if 'Images' not in [s.keys()[0] for s in msg_names]:
                msg_names.append({'Images' : 'images'})

        self.msg_names = []
        assert len(shaders) == len(msg_names)
        for i in range(len(shaders)):
            assert len(shaders[i].keys()) == 1
            for k in shaders[i]:
                assert k in msg_names[i]
                self.msg_names.append(msg_names[i][k])
        self.msg_names = [self.msg_names] * n_cameras
        self.environment_pid = None
        self.num_frames_per_msg = 1 + 1 + n_cameras * len(shaders)
        ctx = zmq.Context()
        self.tc = self.init_tdw_client()
        self.not_yet_joined = True
        #query comm particulars
        self.rng = np.random.RandomState(random_seed)
        self.CACHE = {}
        self.COMPLEXITY = 1500#I think this is irrelevant, or it should be. TODO check
        self.NUM_LIGHTS = 4
        self.ROOM_LENGTH, self.ROOM_WIDTH = room_dims
        self.SCREEN_HEIGHT, self.SCREEN_WIDTH = screen_dims
        self.gpu_num = str(gpu_num)
        #assert self.gpu_num in ['0', '1', '2', '3', '4', '5'], self.gpu_num
        self.timeout = environment_timeout
        self.grouping = None


    def init_tdw_client(self):
        return TDW_Client(self.host_address,
            initial_command='request_create_environment',
            description="test script",
            selected_build=self.selected_build,  # or skip to select from UI
            #queue_port_num="23402",
            get_obj_data=True,
            send_scene_info=False,
            num_frames_per_msg=self.num_frames_per_msg,
            shaders=self.shaders,
            )

    def get_items(self, rng, q, num_items, scale, mass, var = .01, shape_pool = None, color = 0):
        for _k in default_keys:
            if _k not in q:
                q[_k] = {'$exists': True}
        #TODO: do away with this horrific hack in case shape_pool is specified
        #might want to just initialize this once
        query_res = old_assets_query_result_example.query_res
        query_unity_data = query_results_to_unity_data(query_res, \
            scale, mass, var = var, seed = self.unity_seed + 1)
        assert shape_pool is not None#changing from previous version, information must be passed in this way.
        for qu_data in query_unity_data:
            shape_this_time = rng.choice(shape_pool)
            qu_data['aws_address'] = 'PhysXResources/StandardShapes/Solids' + color + '/' + shape_this_time + '.prefab'
        return query_unity_data

    def reset(self, * round_info):
        self.nn_frames = 0
        self.round_info = round_info
        rounds = [{'items' : self.get_items(self.rng, query_dict[info['type']], info['num_items'] * 4, info['scale'], info['mass'], info['scale_var'], shape_pool = info.get('shape_pool'), color = info.get('color', '0')), 'num_items' : info['num_items']} for info in round_info]

        self.config = {
            "environment_scene" : "ProceduralGeneration",
            "random_seed": self.unity_seed, #Omit and it will just choose one at random. Chosen seeds are output into the log(under warning or log level).
            "should_use_standardized_size": False,
            "standardized_size": [1.0, 1.0, 1.0],
            "complexity": self.COMPLEXITY,
            "random_materials": True,
            "num_ceiling_lights": self.NUM_LIGHTS,
            "intensity_ceiling_lights": 1,
            "use_standard_shader": True,
            "minimum_stacking_base_objects": 5,
            "minimum_objects_to_stack": 5,
            "disable_rand_stacking": 0,
            "room_width": self.ROOM_WIDTH,
            "room_height": 10.0,
            "room_length": self.ROOM_LENGTH,
            "wall_width": 1.0,
            "door_width": 1.5,
            "door_height": 3.0,
            "window_size_width": (5.0/1.618), # standard window ratio is 1:1.618
            "window_size_height": 5.0,
            "window_placement_height": 2.5,
            "window_spacing": 7.0,  #Average spacing between windows on walls
            "wall_trim_height": 0.5,
            "wall_trim_thickness": 0.01,
            "min_hallway_width": 5.0,
            "number_rooms": 1,
            "max_wall_twists": 3,
            "max_placement_attempts": 300,   #Maximum number of failed placements before we consider a room fully filled.
            "grid_size": 0.4,    #Determines how fine tuned a grid the objects are placed on during Proc. Gen. Smaller the number, the
            "use_mongodb_inter": 1, 
            'rounds' : rounds
            }
        if self.not_yet_joined:
            self.tc.load_config(self.config)
            self.tc.load_profile({'screen_width': self.SCREEN_WIDTH, 'screen_height': self.SCREEN_HEIGHT, 'gpu_num' : self.gpu_num})
            msg = {'msg' : {}}
            self.sock = self.tc.run()
            self.sock.setsockopt(zmq.LINGER, 0)
            self.poller = zmq.Poller()
            self.poller.register(self.sock, zmq.POLLIN)
            self.not_yet_joined = False
        else:
            print('switching scene...')
            scene_switch_msg = {"msg_type" : "SCENE_SWITCH", "config" : self.config, "get_obj_data" : True, "send_scene_info" : True, 'SHADERS' : self.shaders}
            msg = {"n": self.num_frames_per_msg, "msg": scene_switch_msg}
        return self._observe_world()


    def write_image_to_path(self, image, num, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        Image.fromarray(image).save(
                os.path.join(base_dir, "%06d.png" % num))

    def _observe_world(self):
        self.nn_frames += 1
        # try to receive a message, if no message can be received within
        # the timeout period or the message is errornous close the 
        # current environment and restart a new one
        try:
            self.observation = self.handle_message_new(self.msg_names, 
                timeout=self.timeout)
            if hasattr(self, 'image_dir') and self.image_dir:
                assert 'images1' in self.observation
                self.write_image_to_path(self.observation['images1'], 
                        self.nn_frames, self.image_dir)
            self.observation['info'] = json.loads(self.observation['info'])
            #parse dict to list
            for key in ['avatar_position', 'avatar_up', 'avatar_forward',
                    'avatar_right', 'avatar_velocity', 'avatar_angvel',
                    'avatar_rotation']:
                rec = json.loads(self.observation['info'][key])
                self.observation['info'][key] = [rec['x'], rec['y'], rec['z']]
            for obj in self.observation['info']['observed_objects']:
                for i in [2,3,6,7]:
                    rec = json.loads(obj[i])
                    if i == 3:
                        obj[i] = [rec['x'], rec['y'], rec['z'], rec['w']]
                    else:
                        obj[i] = [rec['x'], rec['y'], rec['z']]
            self.environment_pid = self.observation['info']['environment_pid']
        except IOError as e:
            print('Current environment is broken. Starting new one...')
            print "I/O error: {}".format(e)
            self.tc.close()
            self.tc = self.init_tdw_client()
            self.not_yet_joined = True
            print('...New environment started.')
            if self.environment_pid is not None:
                try:
                    os.kill(self.environment_pid, signal.SIGKILL)
                    print('Killed old environment with pid %d.' \
                            % self.environment_pid)
                except:
                    print('Could not kill old environment with pid %d. Already dead?' \
                            % self.environment_pid)
            raise IOError("Environment restarted, provide new config")
        self.obs_raw = self.observation
        return self.observation

    def handle_message_new(self, msg_names, write = False, 
                outdir = '', imtype =  'png', prefix = '', timeout=None):
        if timeout is None:
            info = self.sock.recv()
        else:
            if self.poller.poll(timeout * 1000): # timeout in seconds
                info = self.sock.recv()
            else:
                raise IOError('Did not receive message within timeout')
        data = {'info': info}
        if self.poller.poll(timeout * 1000):
            data['particles'] = np.reshape(np.fromstring(self.sock.recv(), dtype=np.float32), [-1, 7])
        else:
            raise IOError('Did not receive message within timeout')
        # Iterate over all cameras
        for cam in range(len(msg_names)):
            for n in range(len(msg_names[cam])):
                # Handle set of images per camera
                if timeout is None:
                    imgstr = self.sock.recv()
                else:
                    if self.poller.poll(timeout * 1000): # timeout in seconds
                        imgstr = self.sock.recv()
                    else:
                        raise IOError('Did not receive message within timeout')
                imgarray = np.asarray(Image.open(StringIO(imgstr)).convert('RGB'))
                field_name = msg_names[cam][n] + str(cam+1)
                assert field_name not in data, \
                        ('duplicate message name %s' % field_name)
                data[field_name] = imgarray
        return data

    def _termination_condition(self):
        return False



    def step(self, action):
        #gets message. action_to_message_fn can make adjustments to actio
        #other data is included so that we can manage a cache of data all in one place, but the environment otherwise does not interact with it
        msg = action
        self.sock.send_json(msg)
        if not hasattr(self, "start"):
            self.nn_frames = 0
            self.start = time.time()
        self.observation = self._observe_world()
        term_signal = self._termination_condition()
        return self.observation, term_signal

class TelekineticMagicianEnvironment(TDWClientEnvironment):
    def __init__(self,
        limits,
        max_interaction_distance,
        wall_safety,
        do_torque,
        **kwargs):
        limits = np.array(limits)
        for k_ in ['max_interaction_distance', 'wall_safety', 'do_torque', 'limits']:
            setattr(self, k_, eval(k_))
        self.single_action_len = 6 if self.do_torque else 3
        super(TelekineticMagicianEnvironment, self).__init__(**kwargs)

    def _termination_condition(self):
        safety = .5
        available_objects = [o for o in self.obs_raw['info']['observed_objects'] if int(o[1]) != -1 and not o[4]]
        num_items = sum([info_['num_items'] for info_ in self.round_info])

        if len(available_objects) != num_items:
            print('requesting restart because object dropped out')
            return True
        things_to_check = [(obj_[0], np.array(obj_[2])) for obj_ in available_objects] + [('agent', np.array(self.obs_raw['info']['avatar_position']))]
        for desc, loc in things_to_check:
            for dim, bound in [(0, self.ROOM_WIDTH), (2, self.ROOM_LENGTH)]:
                if loc[dim] < -safety or loc[dim] > bound + safety:
                    print('requesting restart because ' + desc + ' horizontally oob')
                    return True
            if loc[1] < -safety or loc[1] > bound + safety:
                print('requesting restart because ' + desc + ' vertically oob')
                return True
        return False

    def _observe_world(self):
        super(TelekineticMagicianEnvironment, self)._observe_world()
        #avatar position details
        self._avatar_position = np.array(self.observation['info']['avatar_position'])
        self._avatar_forward = np.array(self.observation['info']['avatar_forward'])
        #compute distances and which objects are in play
        oarray = self.observation['objects1']
        oarray1 = 256**2 * oarray[:, :, 0] + 256 * oarray[:, :, 1] + oarray[:, :, 2]
        available_objects = [o for o in self.observation['info']['observed_objects'] if int(o[1]) != -1 and not o[4]]
        self.num_obj = len(available_objects)
        object_distances = [np.linalg.norm(self._avatar_position - np.array(obj_[2])) for obj_ in available_objects]
        close_objects = [obj_ for obj_, d in zip(available_objects, object_distances) if d < self.max_interaction_distance]
        obj_pixels = [(oarray1 == obj_[1]).nonzero() for obj_ in close_objects]
        in_view_close_objects = [(obj_, (xs, ys)) for (obj_, (xs, ys)) in zip(close_objects, obj_pixels) if len(xs) > 0]
        close_centroids = [(obj_, np.array(zip(xs, ys)).mean(0)) for (obj_, (xs, ys)) in in_view_close_objects]
        self._ordered_objs_centroids = sorted(close_centroids, key = lambda (obj_, (xc, yc)) : xc)
        lexi_ordered_objects = sorted(available_objects, key = lambda obj_ : obj_[0])
        distances_of_objects = [np.linalg.norm(np.array(lexi_ordered_objects[i][2]) - np.array(lexi_ordered_objects[j][2]))
                           for i in range(len(available_objects)) for j in range(i + 1, len(available_objects))]
        self.observation = {'obj_there' : len(in_view_close_objects), 'distances_between_objects' : distances_of_objects,\
                            'images1' : self.observation['images1'],
                             'one_hot_obj_there' : [1 if j_ == len(in_view_close_objects) - 1 else 0 for j_ in range(len(available_objects))]}
        return self.observation

    def _ego_helper(self, action, msg):
        agent_vel = action[0]
        wall_feedback = 1.
        proposed_next_position = self._avatar_position + agent_vel * self._avatar_forward
        if proposed_next_position[0] < self.wall_safety or\
           proposed_next_position[0] > self.ROOM_WIDTH - .5 - self.wall_safety or\
           proposed_next_position[2] < self.wall_safety or\
           proposed_next_position[2] > self.ROOM_LENGTH - .5 - self.wall_safety:
            agent_vel = 0.
            wall_feedback = 0.
        msg['msg']['vel'] = [0,0, agent_vel]
        msg['msg']['ang_vel'] = [0, action[1], 0]
        return wall_feedback

    def _object_interaction_helper(self, action, msg):
        num_interacted = len(self._ordered_objs_centroids)
        msg['msg']['action_type'] = 'ACT_' + str(num_interacted) if num_interacted else 'NO_ACT'
        for slot_num, (obj_, cm) in enumerate(self._ordered_objs_centroids):
            msg_action = {'use_absolute_coordinates' : False}
            msg_action['force'] = list(action[2 + slot_num * self.single_action_len: 2 + slot_num * self.single_action_len + 3])
            if self.do_torque:
                msg_action['torque'] = list(action[2 + slot_num * self.single_action_len + 3 : 2 + (slot_num + 1) * self.single_action_len])
            else:
                msg_action['torque'] = [0., 0., 0.]
            msg_action['id'] = str(obj_[1])
            msg_action['object'] = str(obj_[1])
            msg_action['action_pos'] = list(map(float, cm))
            msg['msg']['actions'].append(msg_action)
        return [1. if slot_ < num_interacted else 0. for slot_ in range(self.num_obj)]
            

    def step(self, action):
        msg = init_msg(self.num_frames_per_msg)
        masked_action = action
        action = self.limits * action
        wall_feedback = self._ego_helper(action, msg)
        obj_there_mask = self._object_interaction_helper(action, msg)
        self.observation, term_signal = super(TelekineticMagicianEnvironment, self).step(msg)
        self.observation['wall_feedback'] = wall_feedback
        masked_action[0] = masked_action[0] * wall_feedback
        for (slot_num, obj_there_indicator) in enumerate(obj_there_mask):
            masked_action[2 + slot_num * self.single_action_len : 2 + (slot_num + 1) * self.single_action_len] *= float(obj_there_indicator)
        self.observation['action_post'] = masked_action
        return self.observation, term_signal


