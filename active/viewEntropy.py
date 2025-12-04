import torch
import numpy as np
from PIL import Image
from icecream import ic
import json
import argparse
from glob import glob
from tqdm import tqdm



class Entropy(torch.nn.Module):
  """
  Computes the entropy of the input tensor:
  $H(x) = sum_i (p_i log_2(p_i) )$ where $p_i$ is the input tensor.
  """
  def __init__(self,
               dim = None,
               keepdim : bool = False,
               normalize_input: bool = True,
               normalize_output: bool = True):
    """
    Creates the entropy loss
    $H(x) = sum_i (p_i log_2(p_i) )$ where $p_i$ is the input tensor.
    The summation dimension is controlled by 'dim' and 'keepdim', see torch.sum for
    details.
    :param dim: the dimension for the summation or None if summed over all axes
    :param keepdim: whether the output tensor has 'dim' retained or not.
    :param normalize_input: should the input be normalized to a proper probability
      distribution beforehand (default: true). Set to false to optimize
      the performance if the input is already a probability distribution
    :param normalize_output: should the output be normalized by the size of the
      tensors? This introduces a scaling factor of 1/log_2(N) where N is the
      number of elements in the reduced dimensions.
      This makes this loss independent of the image resolution.
    """
    super().__init__()
    self._dim = dim
    self._keepdim = keepdim
    self._normalize_input = normalize_input
    self._normalize_output = normalize_output

  def forward(self, x):
    if self._dim is None:
      dim = tuple(range(len(x.shape)))
    else:
      dim = self._dim
    # fetch input probability distribution
    if self._normalize_input:
      scale = torch.sum(x, dim, keepdim=True)
      p = x / scale
    else:
      p = x

    # compute entropy
    entropy = -torch.nansum(p * torch.log2(p), dim, keepdim=self._keepdim)
    # entropy = -torch.sum(pyrenderer.mul_log(p), dim, keepdim=self._keepdim)

    # normalize output
    if self._normalize_output:
      N = np.prod([x.shape[i] for i in dim])
      entropy = entropy / np.log2(N)
    # done
    return entropy

class ColorMatches(torch.nn.Module):
  """
  Given an array of N reference colors,
  compute for each pixel the closest reference color
  (softmax on the color difference), then accumulate the area of each region.

  The output will be a probability density function, no
  normalization is needed in the entropy computation following it.

  TODO: different color spaces
  """

  def __init__(self, peaks : np.ndarray):
    """
    Constructs the color matching loss
    :param peaks: the peaks as an array of shape N*3
    """
    super().__init__()
    assert len(peaks.shape)==2
    assert peaks.shape[1] >= 3
    peaks = np.concatenate(
      (np.array([[0,0,0]], dtype=peaks.dtype), peaks[:,:3]), axis=0)
    self._N = peaks.shape[0]

    # B=1 x H=1 x W=1 x C=3 x N
    self.register_buffer("_peaks", torch.from_numpy(peaks.transpose()) \
      .unsqueeze(0).unsqueeze(0).unsqueeze(0))

  def forward(self, x):
    """
    :param x: input image of shape B*H*W*4 (opacity channel is ignored if existing)
    :return:
    """
    B, H, W, C = x.shape
    if C==4:
      x = x[..., :3] # strip opacity

    # compute differences:
    differences = x.unsqueeze(-1) - self._peaks # B*H*W*C*N
    scores = -torch.linalg.norm(differences, dim=3) # B*H*W*N
    scores = torch.nn.functional.softmax(scores, dim=3)
    # assemble pixel region sizes
    Ai = torch.sum(scores, dim=(1,2))
    pi = Ai / (H*W) # normalize, a probability distribution per batch

    return pi


class EntropyLosses(torch.nn.Module):
        def __init__(self, opacity_weight=1, color_weight=0, peak_colors=None):
            super().__init__()

            self._opacity_entropy = Entropy(dim=(1, 2), normalize_input=True, normalize_output=True)
            self._opacity_weight = opacity_weight

            if color_weight>0:
                self._color_matches = ColorMatches(peak_colors)
                self._color_entropy = Entropy(dim=(1,), normalize_input=False)
            self._color_weight = color_weight

        def forward(self, x):
            losses = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

            if self._opacity_weight > 0:
                opacity = x[:, :, :, 3]
                opacity_entropy = self._opacity_entropy(opacity)
                assert losses.shape == opacity_entropy.shape
                losses = losses + self._opacity_weight * opacity_entropy

            if self._color_weight > 0:
                color_regions = self._color_matches(x)
                color_entropy = self._color_entropy(color_regions)
                assert losses.shape == color_entropy.shape
                losses = losses + self._color_weight * color_entropy

            # TODO: measure of the stability
            # maybe some form of summed per-pixel gradients for yaw+pitch
            # how does this work in an adjoint framework?

            return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measuring the view entropy',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--viewDir', type=str, default="./TestViews/combustion", help='One reference image path or dir')
    args = parser.parse_args()
    viewDir = args.viewDir
    
    pngFiles = sorted(glob(viewDir + "/*.png"))
    highestEntropy = -1e9
    highestEntropyPath = ""
    lowestEntropy = 1e9
    lowestEntropyPath = ""
    entropy_scores = [] #*TO Siyuan: This is the final entropy score for each view image following order 0->180
    
    #TF for color entopy
    TF_BASE_DIR = "./TFs/"
    # TF_json_path = TF_BASE_DIR + "vortex.json"
    TF_json_path = TF_BASE_DIR + "combustion.json"
    TF_json = json.load(open(TF_json_path))
    PeakColors = torch.tensor(TF_json[0]["RGBPoints"]).reshape(-1, 4)[:, 1:]
    PeakColors = PeakColors.numpy()
    loss = EntropyLosses(color_weight=1, peak_colors=PeakColors)
    
    for imgPath in tqdm(pngFiles):
        img = Image.open(imgPath)
        img = np.array(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img /= 255.0
        eval_entropy = loss(img)
        entropy_scores.append(eval_entropy.item())
        if eval_entropy > highestEntropy:
            highestEntropy = eval_entropy
            highestEntropyPath = imgPath
        if eval_entropy < lowestEntropy:
            lowestEntropy = eval_entropy
            lowestEntropyPath = imgPath
    print(f"highestEntropy: {highestEntropy}, highestEntropyPath: {highestEntropyPath}")
    print(f"lowestEntropy: {lowestEntropy}, lowestEntropyPath: {lowestEntropyPath}")
    # print(f"entropy_scores: {entropy_scores}")

