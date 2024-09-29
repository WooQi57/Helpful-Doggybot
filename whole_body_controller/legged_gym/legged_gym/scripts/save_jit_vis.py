import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        actor_layers0 = []
        actor_layers0.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers0.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers0.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers0.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers0.append(activation)
        if tanh_encoder_output:
            actor_layers0.append(nn.Tanh())

        self.use_2ac = False
        self.depth_actor_use_actor1 = False
        if self.use_2ac:
            actor_layers1 = []
            actor_layers1.append(nn.Linear(num_prop+
                                        self.scan_encoder_output_dim+
                                        num_priv_explicit+
                                        priv_encoder_output_dim, 
                                        actor_hidden_dims[0]))
            actor_layers1.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers1.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    actor_layers1.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers1.append(activation)
            if tanh_encoder_output:
                actor_layers1.append(nn.Tanh())
            # self.actor_backbone = nn.Sequential(*actor_layers)
            self.actor_backbone0 = nn.Sequential(*actor_layers0)
            self.actor_backbone1 = nn.Sequential(*actor_layers1)
        elif self.depth_actor_use_actor1:
            actor_layers1 = []
            actor_layers1.append(nn.Linear(num_prop+
                                        self.scan_encoder_output_dim+
                                        num_priv_explicit+
                                        priv_encoder_output_dim, 
                                        actor_hidden_dims[0]))
            actor_layers1.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers1.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    actor_layers1.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers1.append(activation)
            if tanh_encoder_output:
                actor_layers1.append(nn.Tanh())
            # self.actor_backbone = nn.Sequential(*actor_layers)
            self.actor_backbone1 = nn.Sequential(*actor_layers1)
        else:
            self.actor_backbone0 = nn.Sequential(*actor_layers0)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        # if not eval:
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
            if scandots_latent is None:
                scan_latent = self.scan_encoder(obs_scan)   
            else:
                scan_latent = scandots_latent
            obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
        obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        # if self.use_2ac:
        #     flat_obs = backbone_input[backbone_input[:, 0] == 0]
        #     step_obs = backbone_input[backbone_input[:, 0] == 1]
        #     backbone_output0 = self.actor_backbone0(flat_obs)
        #     backbone_output1 = self.actor_backbone1(step_obs)

        #     backbone_output = torch.cat([backbone_output0, backbone_output1], dim=0)
        #     backbone_output[backbone_input[:, 0] == 0, :] = backbone_output0
        #     backbone_output[backbone_input[:, 0] == 1, :] = backbone_output1
        #     # print("using 2 ac")
        #     # print(f"{backbone_input=}\n{flat_obs=}\n{step_obs=}\n{backbone_output0=}\n{backbone_output1=}\n{backbone_output=}")
        if self.depth_actor_use_actor1:
            backbone_output = self.actor_backbone1(backbone_input)
            # print("using actor1")
        else:
            backbone_output = self.actor_backbone0(backbone_input)
            # print("using actor0")
        return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)
        self.actor.use_2ac = False

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs, depth_latent):
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)
        # return obs, depth_latent

def play(args):    
    load_run = "../../logs/doggybot/" + args.exptid
    checkpoint = args.checkpoint

    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 4 + 1 + 14+14#12 +12
    num_scan = 132
    num_actions = 13 #12
    
    # depth_buffer_len = 2
    depth_resized = (87, 58)
    
    n_proprio = 1+3 + 2 + 1 + 5 + 13*3 + 4 #3 + 2 + 3 + 4 + 36 + 4 +1
    history_len = 10

    device = torch.device('cpu')
    policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    # policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    
    policy = policy.to(device)#.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}
    torch.save(state_dict, os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 1
        
        obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
        depth_latent = torch.ones(1, 32, device=device)
        test = policy(obs_input,depth_latent)
        
        traced_policy = torch.jit.trace(policy, (obs_input,depth_latent))
        
        # traced_policy = torch.jit.script(policy)
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(os.path.join(load_run, "traced")))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
    