# -*- coding: utf8 -*-
#

import torch

output = torch.randn(2, 32, 128)

indices = torch.tensor([
    [1, 2],
    [5, 8]
])

mask_output = output.gather(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, output.size(-1)))

print((mask_output[0][0] == output[0][1]).all())
print((mask_output[0][1] == output[0][2]).all())

print((mask_output[1][0] == output[1][5]).all())
print((mask_output[1][1] == output[1][8]).all())
