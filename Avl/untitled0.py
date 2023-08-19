# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:53:32 2023

@author: Plutonium
"""

import torch

a = torch.rand(4,2,3)
print(a)

print('!!!!!!!!!!!')
b = a[a[:, :, 0].sort()[1]]
print(b)

c = b[b[:, 0].sort()[1]]
print('???????????????')
print(c)
#%%
ax =  a[:,:,0]
ay =  a[:,:,1]
az =  a[:,:,2]

ax_sort, idx  = ax.sort(dim=0)
print(ax_sort)

#%%
ay_sort = ay[idx]
az_sort = az[idx]

a_sort = torch.cat((ax_sort.unsqueeze(-1), ay_sort.unsqueeze(-1), az_sort.unsqueeze(-1)), dim=2)

#%%
b, idx = a[:,:,0].permute([1,0]).sort(dim=1)


print()
print(a)


print()
print(b)

#%%

a[:,:,0]

#%%

idx

c = a[:,:,0].permute([1,0])


print(c.permute([2,1,0]))

#%%
c.shape
