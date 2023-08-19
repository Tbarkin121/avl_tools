# -*- codng: utf-8 -*-
"""
Created on Mon Aug 14 16:53:02 2023

@author: MiloPC
"""

import torch
import vlmlib
import matplotlib.pyplot as plt
import time
import os

I = 50
J = 1

alphas = torch.linspace(-5.0, 5.0, 10)
# b = torch.linspace(-0.1, 0.1, J)
betas = torch.linspace(0.0, 0.0, J)

wt_pos = torch.linspace(0.0, 1.0, 10)
# wt_pos = torch.tensor([0.2])
mass_vals = torch.linspace(0.00, 0.1, 10)
now = time.perf_counter()

directory  = 'data\\'

curr_case = 0
total_cases = len(wt_pos) * len(mass_vals) * len(alphas)

for alp in alphas:
    for m in mass_vals:
        for wt in wt_pos:
    
            print('\nCase: ',curr_case,'/',total_cases,'\n')
            curr_case += 1
            
            # filename = f'_Atail_{i}_{j}'
            filename = directory + f'_Atail_mass{m:0.3f}_tippos{wt:0.2f}_a{alp:02f}'
            # filename = '_Atail'
            surface_names = ['Vibrant_Dragonscale_Wing',
                              'Lost_Ancient_God_Killer_Tail']
            
            dist = torch.tensor([[10, -2,  20, -2],
                                  [10,  1,  15,  1]])
            
            y_mirror =  [[True, 0.0],
                          [True, 0.0]]
            
            angle =    torch.tensor([0.0,
                                    -5.0])
            
            translate = [torch.tensor([0.000, 0.000, 0.000]),
                          torch.tensor([0.405, 0.000, 0.060])]
                        
            sections0 = [torch.tensor([[0.00,  0.000, 0.000, 0.150, 0.0, 4412],
                                        [0.00,  0.055, 0.000, 0.150, 0.0, 4412],
                                        [0.04,  0.135, 0.000, 0.110, 0.0, 4412],
                                        [wt,   0.435, 0.025, 0.060, 0.0, 4412]]),
                        
                          torch.tensor([[0.0,   0.00,  0.00,  0.05,  0.0, 12],   #should be 0012
                                        [0.0,   0.06, -0.06,  0.05,  0.0, 12]])] #should be 0012
            
            ctrl = [[[],
                      [],
                      ['aeleron',torch.tensor([1.0,  0.25,  0.0, 1.0, 0.0, 1.0 ])],
                      ['aeleron',torch.tensor([1.0,  0.25,  0.0, 1.0, 0.0, 1.0 ])]],
                    [['ruddervator',torch.tensor([1.0,  0.25,  0.0, 0.7071, -0.7071, 1.0 ])],
                      ['ruddervator',torch.tensor([1.0,  0.25,  0.0, 0.7071, -0.7071, 1.0 ])]]]
            
            
            mass_point = torch.tensor([[m]])
            xyzpoint = torch.tensor([[0.030,   0.0,  -0.002]])
            rho_area =  torch.tensor([0.55])
            
            sections = vlmlib.duplicateSections(sections0,y_mirror)
            sections_translated = vlmlib.translateSections(sections,translate)
            quads = vlmlib.sections2quads(sections_translated)
            area_surf, CoM_surf = vlmlib.quadareas(quads)
            mass_surf = area_surf*rho_area
            mass = torch.cat([mass_surf,mass_point],dim=0)
            mass_tot = mass.sum()
            CoM = torch.cat([CoM_surf,xyzpoint],dim=0)
            area_surf_tot = area_surf.sum()
            CoM_tot = (CoM*mass).sum(dim=0)/mass_tot
            xyz_c = quads - CoM_tot.unsqueeze(dim=0).unsqueeze(dim=0)
            
            xyzpoint_c = xyzpoint - CoM_tot.unsqueeze(dim=0)
            
            I_point_tot = vlmlib.pointmoi(xyzpoint_c, mass_point)
            I_surf_tot = rho_area*vlmlib.quadmois(xyz_c)
            I_tot = I_point_tot + I_surf_tot
            
            #%%
            avg_chord = 0.08
            span = 0.890
            cdcl = torch.tensor([[-0.276, 0.033, 0.551, 0.016, 1.379, 0.0260],
                                 [-0.968, 0.048, 0.000, 0.017, 0.968, 0.048]])
            in_avl = [filename,
                      surface_names,
                      dist,
                      y_mirror,
                      angle,
                      translate,
                      sections0,
                      ctrl,
                      area_surf_tot,
                      avg_chord,
                      span,
                      cdcl]
            
            vlmlib.makeMassFile(filename,mass_tot,CoM_tot,I_tot)
            vlmlib.makeAVLfile2(in_avl)
            
            alpha = float(alp)
            beta = 0
            
            vlmlib.run_AVL2(filename,alpha,beta)

dt = time.perf_counter() - now

print(dt)
# for i in range(I):
#     for j in range(J):
        

#         print('\nCase: ',i*J+j + 1,'/',I*J,'\n')
#         # filename = f'_Atail_{i}_{j}'
#         filename = directory + f'_Atail_{a[i].cpu().numpy():0.2f}_{a[j].cpu().numpy():0.2f}'
#         # filename = '_Atail'
#         surface_names = ['Vibrant_Dragonscale_Wing',
#                          'Lost_Ancient_God_Killer_Tail']
        
#         dist = torch.tensor([[10, -2,  20, -2],
#                              [10,  1,  15,  1]])
        
#         y_mirror =  [[True, 0.0],
#                      [True, 0.0]]
        
#         angle =    torch.tensor([0.0,
#                                 -5.0])
        
#         translate = [torch.tensor([0.000, 0.000, 0.000]),
#                      torch.tensor([0.405, 0.000, 0.060])]
                    
#         sections0 = [torch.tensor([[0.00,  0.000, 0.000, 0.150, 0.0, 4412],
#                                    [0.00,  0.055, 0.000, 0.150, 0.0, 4412],
#                                    [0.04,  0.135, 0.000, 0.110, 0.0, 4412],
#                                    [0.2,   0.435, 0.025, 0.060, 0.0, 4412]]),
                    
#                      torch.tensor([[0.0,   0.00,  0.00,  0.05,  0.0, 4412],
#                                    [0.0,   0.06, -0.06,  0.05,  0.0, 4412]])]
        
#         ctrl = [[[],
#                  [],
#                  ['aeleron',torch.tensor([1.0,  0.25,  0.0, 1.0, 0.0, 1.0 ])],
#                  ['aeleron',torch.tensor([1.0,  0.25,  0.0, 1.0, 0.0, 1.0 ])]],
#                 [['ruddervator',torch.tensor([1.0,  0.25,  0.0, 0.7071, -0.7071, 1.0 ])],
#                  ['ruddervator',torch.tensor([1.0,  0.25,  0.0, 0.7071, -0.7071, 1.0 ])]]]
        
        
#         mass_point = torch.tensor([[0.030]])
#         xyzpoint = torch.tensor([[0.030,   0.0,  -0.002]])
#         rho_area =  torch.tensor([0.55])
        
    
        
        
        
#         #%%
        
#         sections = vlmlib.duplicateSections(sections0,y_mirror)
#         sections_translated = vlmlib.translateSections(sections,translate)
#         quads = vlmlib.sections2quads(sections_translated)
#         area_surf, CoM_surf = vlmlib.quadareas(quads)
#         mass_surf = area_surf*rho_area
#         mass = torch.cat([mass_surf,mass_point],dim=0)
#         mass_tot = mass.sum()
#         CoM = torch.cat([CoM_surf,xyzpoint],dim=0)
#         area_surf_tot = area_surf.sum()
#         CoM_tot = (CoM*mass).sum(dim=0)/mass_tot
#         xyz_c = quads - CoM_tot.unsqueeze(dim=0).unsqueeze(dim=0)
        
#         xyzpoint_c = xyzpoint - CoM_tot.unsqueeze(dim=0)
        
#         I_point_tot = vlmlib.pointmoi(xyzpoint_c, mass_point)
#         I_surf_tot = rho_area*vlmlib.quadmois(xyz_c)
#         I_tot = I_point_tot + I_surf_tot
        
#         #%%
        
#         in_avl = [filename,
#                   surface_names,
#                   dist,
#                   y_mirror,
#                   angle,
#                   translate,
#                   sections0,
#                   ctrl,
#                   area_surf_tot]
        
#         vlmlib.makeMassFile(filename,mass_tot,CoM_tot,I_tot)
#         vlmlib.makeAVLfile2(in_avl)
        
#         alpha = a[i].cpu().numpy()
#         beta = b[j].cpu().numpy()
        
#         vlmlib.run_AVL2(filename,alpha,beta)

# dt = time.perf_counter() - now

# print(dt)




#%%
# xyzplo = torch.cat([quads,quads[:,0,:].unsqueeze(dim=1)],dim=1).cpu().numpy()
# CoMplo = CoM_tot.squeeze().cpu().numpy()

# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# for i in range(quads.shape[0]):
#     ax.plot(xyzplo[i,:,0],xyzplo[i,:,1],xyzplo[i,:,2],'r') 
#     ax.plot(CoM_surf[i,0].cpu().numpy(),CoM_surf[i,1].cpu().numpy(),CoM_surf[i,2].cpu().numpy(),'bx')
# ax.plot(CoMplo[0],CoMplo[1],CoMplo[2],'cx')
# ax.plot(xyzpoint[:,0].cpu().numpy(),xyzpoint[:,1].cpu().numpy(),xyzpoint[:,2].cpu().numpy(),'bo')
# ax.set_aspect('equal')

# print(mass_tot)
# print(CoM_tot)
# print(I_tot)




