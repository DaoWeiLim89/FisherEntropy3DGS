import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from active import viewEntropy

class HRegSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        # Initialize the entropy loss module
        self.entropy_weight = 0.8
        self.entropy_loss = viewEntropy.EntropyLosses(
            opacity_weight=1.0,
            color_weight= 1.0, 
            peak_colors=None # Not sure what this variable is for
        )

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
        # Run heesian on training set
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

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

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

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
            print(f"Image value range: [{pred_img.min().item()}, {pred_img.max().item()}]")
            
            # make sure the image is in [0, 1] range
            img = torch.clamp(img, 0.0, 1.0)

            entropy = self.entropy_loss(img).item()
            entropies.append(entropy)

        entropies = np.array(entropies)

        selected_idxs = []

        for _ in range(num_views):
            I_train = torch.reciprocal(H_train + self.reg_lambda)
            
            if self.I_acq_reg:
                acq_scores = np.array([torch.sum((cur_H + self.reg_lambda) * I_train).item() for cur_H in H_candidates])
            else:
                acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])
            
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

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

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
        if self.I_test == True:
            combined_scores *= -1

        _, indices = torch.sort(combined_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs
