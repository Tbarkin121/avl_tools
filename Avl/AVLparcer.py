# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:31:47 2023

@author: Plutonium
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
#%%

directory  = 'data\\'
st_files = [file for file in os.listdir(directory) if file.endswith('.st')]
mass_files = [file for file in os.listdir(directory) if file.endswith('.mass')]
avl_files = [file for file in os.listdir(directory) if file.endswith('.avl')]
print(st_files)
print(mass_files)
print(avl_files)


#%%
variable_pattern = r'([\w/\'-]+)\s*=\s*([-+]?\d*\.\d+)'
hastag_pattern = r'#\s*(\w+)\s*=\s*([\d.E+-]+)' # (# Surfaces =   4)

st_dicts = []
for file in st_files: 
    with open(directory + file) as f:
        local_dict = {}
        # raw_txt = f.read()
        lines = f.readlines()        
        # values = list(iterate_values(lines))
        
        for line in lines:
            # line_fixed = re.sub(r"'", r"\'", line)
            matches = re.findall(variable_pattern, line)
            for key, value in matches:
                local_dict[key] = float(value)
                
            matches = re.findall(hastag_pattern, line)
            for key, value in matches:
                local_dict[key] = float(value)
    st_dicts.append(local_dict)

mass_dicts = []
for file in mass_files: 
    with open(directory + file) as f:
        local_dict = {}
        # raw_txt = f.read()
        lines = f.readlines()     
        
        
        header_line = None
        values_line = None

        for line in lines:
            # line_fixed = re.sub(r"'", r"\'", line)
            matches = re.findall(variable_pattern, line)
            for key, value in matches:
                local_dict[key] = float(value)
                
            matches = re.findall(hastag_pattern, line)
            for key, value in matches:
                local_dict[key] = float(value)
                
                
            # Looking for inertial data lines... different formatting from other things
            if re.match(r'#\s*\w+', line):
                header_line = line
            elif re.match(r'\d+\.\d+', line):
                values_line = line
            
            if header_line and values_line:
                header = re.findall(r'\b(?!#)\S+\b', header_line)
                values = re.findall(r'[-+]?\d*\.\d+|\d+', values_line)
                for key, value in zip(header, values):
                    local_dict[key] = float(value)

    mass_dicts.append(local_dict)
    
    
    
avl_dicts = []
for file in avl_files: 
    with open(directory + file) as f:
        local_dict = {}
        raw_txt = f.read()
        # lines = f.readlines()        
        
        # Define a regular expression pattern to match data sections
        pattern = r'^(SURFACE|SECTION|NACA|CONTROL|YDUPLICATE|ANGLE|TRANSLATE)\n(.*?)(?=\n(?:\w+\n|$))'
        
        # Find all matches using the pattern
        matches = re.findall(pattern, raw_txt, re.DOTALL | re.MULTILINE)
        
        # Process the matches and create a dictionary

        current_surface = None
        for match in matches:
            print(match)
            section = match[0]
            data_block = match[1].strip().split('\n')
            
            if (section == 'SURFACE'):
                current_surface = data_block[0]
                current_section = 0
                local_dict[current_surface] = {}

            if(current_surface):         
                if (section == 'YDUPLICATE'):
                    local_dict[current_surface][section] = float(data_block[0])
                    
                if (section == 'ANGLE'):
                    local_dict[current_surface][section] = float(data_block[0])
                    
                if (section == 'TRANSLATE'):
                    numbers = re.split(r'\s+', data_block[0].strip())
                    local_dict[current_surface][section] = {}
                    local_dict[current_surface][section]['X'] = float(numbers[0])
                    local_dict[current_surface][section]['Y'] = float(numbers[1])
                    local_dict[current_surface][section]['Z'] = float(numbers[2])
                    
                if (section == 'SECTION'):
                    current_section += 1
                    numbers = re.split(r'\s+', data_block[0].strip())
                    local_dict[current_surface][section + str(current_section)] = {}
                    local_dict[current_surface][section + str(current_section)]['Xle'] = float(numbers[0])
                    local_dict[current_surface][section + str(current_section)]['Yle'] = float(numbers[1])
                    local_dict[current_surface][section + str(current_section)]['Zle'] = float(numbers[2])
                    local_dict[current_surface][section + str(current_section)]['chord'] = float(numbers[3])
                    local_dict[current_surface][section + str(current_section)]['angle'] = float(numbers[4])
                    
                    
                if (section == 'NACA'):
                    local_dict[current_surface]['SECTION' + str(current_section)][section] = float(data_block[0])
                
        avl_dicts.append(local_dict)



#%%
# Alpha Plotting
alpha_list = []
CLtot_list = []
CDtot_list = []
Cmtot_list = []
Cma_list = []
NP_list = []
for d in st_dicts:
    alpha_list.append(d['Alpha'])
    CLtot_list.append(d['CLtot'])
    CDtot_list.append(d['CDtot'])
    Cmtot_list.append(d['Cmtot'])
    Cma_list.append(d['Cma'])
    NP_list.append(d['Xnp'])
    
x_data = np.array(alpha_list)
y_data = np.array(Cma_list)

plt.close('all')
# plt.plot(x_data, y_data)
plt.figure()
plt.scatter(x_data, y_data, color='blue', marker='o', s=20, label='Data Points')
plt.xlabel('Alpha')
plt.ylabel('Cma')
plt.grid(True)
plt.show()


y_data = np.array(Cmtot_list)
plt.figure()
plt.scatter(x_data, y_data, color='blue', marker='o', s=20, label='Data Points')
plt.xlabel('Alpha')
plt.ylabel('Cmtot')
plt.grid(True)
plt.show()

y_data = np.array(NP_list)
plt.figure()
plt.scatter(x_data, y_data, color='blue', marker='o', s=20, label='Data Points')
plt.xlabel('Alpha')
plt.ylabel('Neutral Point (x)')
plt.grid(True)
plt.show()
#%%
# Mass / Wingtip Plotting
alpha_list = []
CLtot_list = []
CDtot_list = []
Cmtot_list = []
Cma_list = []
NP_list = []
for d in st_dicts:
    alpha_list.append(d['Alpha'])
    CLtot_list.append(d['CLtot'])
    CDtot_list.append(d['CDtot'])
    Cmtot_list.append(d['Cmtot'])
    Cma_list.append(d['Cma'])
    NP_list.append(d['Xnp'])

Mass_list = []
CGx_list = []
for d in mass_dicts:
    Mass_list.append(d['mass'])
    CGx_list.append(d['x'])
    
WTx_list = []
for d in avl_dicts:
    WTx_list.append(d['Vibrant_Dragonscale_Wing']['SECTION4']['Xle'])
    

x_data = np.array(Mass_list)
x_data = x_data - np.min(x_data)
y_data = np.array(WTx_list) 
z1_data = np.array(Cmtot_list)
z2_data = np.array(Cma_list)


idx1 = np.where(np.abs(z1_data) < 5e-2)[0]
idx2 = np.where(z2_data < -.1)[0]
common_idx = np.intersect1d(idx1, idx2)

b_x = x_data[np.setdiff1d(idx1, common_idx)]
b_y = y_data[np.setdiff1d(idx1, common_idx)]
b_z = z1_data[np.setdiff1d(idx1, common_idx)]

g_x = x_data[common_idx]
g_y = y_data[common_idx]
g_z = z1_data[common_idx]

r_x = x_data[np.setdiff1d(np.setdiff1d(np.arange(len(x_data)), common_idx), idx1)]
r_y = y_data[np.setdiff1d(np.setdiff1d(np.arange(len(y_data)), common_idx), idx1)]
r_z = z1_data[np.setdiff1d(np.setdiff1d(np.arange(len(z1_data)), common_idx), idx1)]


alp_ch_val = 0.1
plt.close('all')
# plt.plot(x_data, y_data)
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(r_x, r_y, r_z, color='red', marker='o', alpha=alp_ch_val)
ax1.scatter(g_x, g_y, g_z, color='green', marker='o', alpha=alp_ch_val)
ax1.scatter(b_x, b_y, b_z, color='blue', marker='o', alpha=alp_ch_val)
ax1.set_xlabel('Mass Val (g)')
ax1.set_ylabel('Wingtip Pos (m)')
ax1.set_zlabel('CM total')
plt.grid(True)
plt.show()


b_x = x_data[np.setdiff1d(idx2, common_idx)]
b_y = y_data[np.setdiff1d(idx2, common_idx)]
b_z = z2_data[np.setdiff1d(idx2, common_idx)]

g_x = x_data[common_idx]
g_y = y_data[common_idx]
g_z = z2_data[common_idx]

r_x = x_data[np.setdiff1d(np.setdiff1d(np.arange(len(x_data)), common_idx), idx2)]
r_y = y_data[np.setdiff1d(np.setdiff1d(np.arange(len(y_data)), common_idx), idx2)]
r_z = z2_data[np.setdiff1d(np.setdiff1d(np.arange(len(z2_data)), common_idx), idx2)]

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(r_x, r_y, r_z, color='red', marker='o', alpha=alp_ch_val)
ax2.scatter(g_x, g_y, g_z, color='green', marker='o', alpha=alp_ch_val)
ax2.scatter(b_x, b_y, b_z, color='blue', marker='o', alpha=alp_ch_val)
ax2.set_xlabel('Mass Val (g)')
ax2.set_ylabel('Wingtip Pos (m)')
ax2.set_zlabel('Cma')
plt.grid(True)
plt.show()



#%%


