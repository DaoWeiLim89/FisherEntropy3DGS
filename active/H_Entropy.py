import torch
import numpy as np
import json
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import os
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from active import viewEntropy

class HEntropySelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg
        self.entropy_use: bool = args.entropy_use
        self.peak_colors = os.path.normpath(args.source_path) + "/TF.json"

        # Load peak colors from TF.json
        peak_colors_json = None
        peak_colors_array = None
        if os.path.exists(self.peak_colors):
            with open(self.peak_colors, 'r') as f:
                peak_colors_json = json.load(f)
            if "RGBPoints" in peak_colors_json[0]:
                rgb_points = peak_colors_json[0]["RGBPoints"]
                if len(rgb_points) % 4 == 0:
                    peak_colors_list = []
                    for i in range(0, len(rgb_points), 4):
                        # Skip the scalar value at index i, take RGB at i+1, i+2, i+3
                        rgb = [rgb_points[i+1], rgb_points[i+2], rgb_points[i+3]]
                        peak_colors_list.append(rgb)
                    peak_colors_array = np.array(peak_colors_list)
                    print(f"Loaded {len(peak_colors_array)} peak colors from TF.json")
                else:
                    print(f"Warning: RGBPoints has unexpected length {len(rgb_points)} (not divisible by 4)")
        else:
            print(f"Warning: TF.json not found at {self.peak_colors}. Color entropy will not function properly.")

        # Initialize the entropy loss module
        self.entropy_weight = 1.0
        self.entropy_loss = viewEntropy.EntropyLosses(
            opacity_weight=0.0,
            color_weight=1.0, 
            peak_colors=peak_colors_array # To define what colors to focus on
        )
        self.entropy_loss = self.entropy_loss.to("cuda")

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        viewpoint_cams = scene.getTrainCameras().copy()

        if self.I_test == True:
            viewpoint_cams = scene.getTestCameras()

        params = gaussians.capture()[1:7]
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]

        # off load to cpu to avoid oom with greedy algo
        device = params[0].device if num_views == 1 else "cpu"
        # device = "cpu" # we have to load to cpu because of inflation

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)

        candidate_cameras = scene.getCandidateCameras()
        # Run hessian on training set
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) ** 2 for p in params])

            H_train += cur_H

            gaussians.optimizer.zero_grad(set_to_none = True) 

        H_train = H_train.to(device)

        if num_views == 1:
            return self.select_single_view(H_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func)

        H_candidates = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) ** 2 for p in params])

            H_candidates.append(cur_H.to(device))

            gaussians.optimizer.zero_grad(set_to_none = True) 
        
        # Calculate entropy scores
        entropies = []
        for cam in tqdm(candidate_cameras, desc="Calculating view entropies for candidate views"):
            if exit_func():
                raise RuntimeError("csm should exit early")
            
            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"] # Shape: C*H*W
            
            # Reshape to B*H*W*C format that entropy expects
            img = pred_img.permute(1, 2, 0).unsqueeze(0)  # C*H*W -> H*W*C -> 1*H*W*C
            #print(f"Image value range: [{pred_img.min().item()}, {pred_img.max().item()}]")
            
            # make sure the image is in [0, 1] range
            img = torch.clamp(img, 0.0, 1.0)

            entropy = self.entropy_loss(img).item()
            entropies.append(entropy)

        entropies = np.array(entropies)

        selected_idxs = []

        for _ in range(num_views):
            acq_scores = np.array([
                torch.sum(torch.log1p(cur_H / (H_train + self.reg_lambda))).item()
                for cur_H in H_candidates
            ])

            combined_scores = acq_scores + entropies * self.entropy_weight

            selected_idx = combined_scores.argmax()
            selected_idxs.append(candidate_views.pop(selected_idx))

            H_train += H_candidates.pop(selected_idx)

            # Remove selected view's entropy too
            entropies = np.delete(entropies, selected_idx)

        return selected_idxs

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, I_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        I_train = torch.reciprocal(I_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))
        entropies = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) ** 2 for p in params])

            I_acq = cur_H

            if self.I_acq_reg:
                I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()

            # Calculate entropy (need to re-render since we did backward)
            with torch.no_grad():
                render_pkg = modified_render(cam, gaussians, pipe, background)
                pred_img = render_pkg["render"]
                img = pred_img.permute(1, 2, 0).unsqueeze(0)  # C*H*W -> 1*H*W*C
                img = torch.clamp(img, 0.0, 1.0)
                entropies[idx] = self.entropy_loss(img).item()
        
        # Combine scores
        combined_scores = acq_scores + entropies * self.entropy_weight

        print(f"combined_scores: {combined_scores.tolist()}")

        _, indices = torch.sort(combined_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs
