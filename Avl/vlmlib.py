
import torch
import os 
import subprocess
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_device('cuda')

plt.style.use('dark_background')

#%%
def VORVEL_VEC(r,r_a,r_b,LBOUND):
    one_over_4pi = 1/4/torch.pi
   
    n1 = r.shape[1]
    n2 = r_a.shape[2]
   
    x_hat = torch.zeros(1,1,n2,3)
    x_hat[0,0,:,0] = 1     

    a = r - r_a
    b = r - r_b
    
    norm_a = torch.reshape(torch.sqrt(torch.sum(a**2,dim=3)),[1,n1,n2,1])
    norm_b = torch.reshape(torch.sqrt(torch.sum(b**2,dim=3)),[1,n1,n2,1])
    
    a_cross_b = torch.cross(a,b)
    a_dot_b = torch.reshape(torch.sum(a*b,dim=3),[1,n1,n2,1])
    
    a_cross_x_hat = torch.cross(a,x_hat)
    b_cross_x_hat = torch.cross(b,x_hat)
    
    a_dot_x_hat = torch.reshape(torch.sum(a*x_hat,dim=3),[1,n1,n2,1])
    b_dot_x_hat = torch.reshape(torch.sum(b*x_hat,dim=3),[1,n1,n2,1])
    
    T =  (1/norm_a + 1/norm_b)/(norm_a*norm_b + a_dot_b)
    TA = (1/norm_a)/(norm_a - a_dot_x_hat)
    TB = -(1/norm_b)/(norm_b - b_dot_x_hat)
    
    T[torch.isnan(T)] = 0.
    T[torch.isinf(T)] = 0.
    TA[torch.isnan(TA)] = 0.
    TA[torch.isinf(TA)] = 0.
    TB[torch.isnan(TB)] = 0.
    TB[torch.isinf(TB)] = 0.
    
    Vi = one_over_4pi*(a_cross_b*T*LBOUND + a_cross_x_hat*TA + b_cross_x_hat*TB)

    return Vi
#%%
def VORVELC_VEC(r,r_a,r_b, RCORE,LBOUND):
    
    
    n1 = r.shape[1]
    n2 = r_a.shape[2]
    
    one_over_4pi = 1/4/torch.pi
   
    x_hat = torch.zeros(1,1,n2,3)
    x_hat[0,0,:,0] = 1     

    a = r - r_a
    b = r - r_b
    
    ASQ = torch.sum(a**2,dim=3).reshape([1,n1,n2,1])
    BSQ = torch.sum(b**2,dim=3).reshape([1,n1,n2,1])
    
    AXISQ = (a[:,:,:,2]**2 +a[:,:,:,1]**2).unsqueeze(dim=3)
    BXISQ = (b[:,:,:,2]**2 +b[:,:,:,1]**2).unsqueeze(dim=3)
    
    norm_a = ASQ.sqrt()
    norm_b = BSQ.sqrt()
    
    a_cross_b = torch.cross(a,b)
    AXBSQ = torch.sum(a_cross_b**2,dim=3).reshape([1,n1,n2,1])
    a_dot_b = torch.reshape(torch.sum(a*b,dim=3),[1,n1,n2,1])
    
    ALSQ  = ASQ + BSQ - 2.0*a_dot_b
    
    a_cross_x_hat = torch.cross(a,x_hat)
    b_cross_x_hat = torch.cross(b,x_hat)
    
    a_dot_x_hat = torch.reshape(a[:,:,:,0],[1,n1,n2,1])
    b_dot_x_hat = torch.reshape(b[:,:,:,0],[1,n1,n2,1])
    
    T = ((BSQ-a_dot_b)/torch.sqrt(BSQ+RCORE**2) + (ASQ-a_dot_b)/torch.sqrt(ASQ+RCORE**2) ) / (AXBSQ + ALSQ*RCORE**2)
    TA = (1.0 + a_dot_x_hat/norm_a) / (AXISQ + RCORE**2)
    TB = -(1.0 + b_dot_x_hat/norm_b) / (BXISQ + RCORE**2)
    
    T[torch.isnan(T)] = 0.
    T[torch.isinf(T)] = 0.
    TA[torch.isnan(TA)] = 0.
    TA[torch.isinf(TA)] = 0.
    TB[torch.isnan(TB)] = 0.
    TB[torch.isinf(TB)] = 0.
    
    Vi = one_over_4pi*(a_cross_b*T*LBOUND + a_cross_x_hat*TA + b_cross_x_hat*TB)
    
    

    return Vi

#%%
def makeAVLfile(filename,surface_names,nc,y_mirror,translate,sections,ns,sref):

    with open(filename+'.avl', 'w') as f:
    
        f.write(f"{filename}\n")
        f.write(f"00.0                    !   Mach\n\
0   0   0        !   iYsym  iZsym  Zsym\n\
{sref}   0.0   0.0         !   Sref   Cref   Bref   reference area, chord, span\n\
0.0   0.0   0.0         !   Xref   Yref   Zref   moment reference location (arb.)")
        f.write("\n#\n#================================\n#\n")
        for i in range(len(nc)):
            
            nc_i = nc[i]
            ns_i = ns[i].sum()
            tx = translate[i][0].cpu().numpy()
            ty = translate[i][1].cpu().numpy()
            tz = translate[i][2].cpu().numpy()
            angle = 0.0
            surface_name_i = surface_names[i]
            f.write(f"SURFACE\n{surface_name_i}\n# Horshoe Vortex Distribution\n\
    {nc_i}  0.0  {ns_i}  0.0       ! Nchord   Cspace   Nspan   Sspace\n")
    
            if y_mirror[i]:
                f.write("# reflect image wing about y=0 plane\nYDUPLICATE\n0.00000\n")
        
            f.write(f'# twist angle bias for whole surface\nANGLE\n{angle}\n')
            
            f.write(f'# x,y,z bias for whole surface\nTRANSLATE\n{tx} {ty} {tz}\n')
            
            for j in range(sections[i].shape[0]):
                Xle = sections[i][j,0] 
                Yle = sections[i][j,1]
                Zle = sections[i][j,2]
                chord = sections[i][j,3]
                f.write(f'#-----------------------------------------\n\
    #   Xle         Yle         Zle         chord       angle  \n\
    SECTION\n\
        {Xle}         {Yle}         {Zle}         {chord}        0.000\n')
        
def makeAVLfile2(in_avl):

    filename,surface_names,dist,y_mirror,angle,translate,sections,ctrl,sref,cref,bref,cdcl = in_avl

    with open(filename+'.avl', 'w') as f:
    
        f.write(f"{filename}\n")
        f.write(f"00.0                    !   Mach\n\
0     0     0           !   iYsym  iZsym  Zsym\n\
{sref}   {cref}   {bref}         !   Sref   Cref   Bref   reference area, chord, span\n\
0.0   0.0   0.0         !   Xref   Yref   Zref   moment reference location (arb.)")
        
        for i in range(len(surface_names)):
            
            nc_i = dist[i][0]
            cs_i = dist[i][1]
            ns_i = dist[i][2]
            ss_i = dist[i][3]
            
            tx = translate[i][0].cpu().numpy()
            ty = translate[i][1].cpu().numpy()
            tz = translate[i][2].cpu().numpy()
            ang = angle[i]
            surface_name_i = surface_names[i]
            
            clmin = float(cdcl[i][0])
            cdmin = float(cdcl[i][1])
            clz = float(cdcl[i][2])
            cdz = float(cdcl[i][3])
            clmax = float(cdcl[i][4])
            cdmax = float(cdcl[i][5])
            f.write("\n#\n#===================================================\n#\n\n")
            f.write(f"SURFACE\n{surface_name_i}\n")
            f.write('#-----------------------------------------\n')
            f.write(f"# Horshoe Vortex Distribution\n\
{nc_i:.0f}  {cs_i:.4f}  {ns_i:.0f}  {ss_i:.4f}       ! Nchord   Cspace   Nspan   Sspace\n")
            f.write('#-----------------------------------------\n')
            if y_mirror[i][0]:
                f.write(f'# reflect image wing about y=0 plane\nYDUPLICATE\n{y_mirror[i][1]:.4f}\n')
            f.write('#-----------------------------------------\n')
            f.write(f'# twist angle bias for whole surface\nANGLE\n{ang:.4f}\n')
            f.write('#-----------------------------------------\n')
            f.write(f'# x,y,z bias for whole surface\nTRANSLATE\n{tx:.4f} {ty:.4f} {tz:.4f}\n')
            f.write('#-----------------------------------------\n')
            f.write(f'#CLMIN CDMIN  CLZERO CDZERO  CLMAX CDMAX\nCDCL\n{clmin:.4f} {cdmin:.4f} {clz:.4f} {cdz:.4f} {clmax:.4f} {cdmax:.4f}\n')
            
            for j in range(sections[i].shape[0]):
                Xle = sections[i][j,0] 
                Yle = sections[i][j,1]
                Zle = sections[i][j,2]
                chord = sections[i][j,3]
                twist = sections[i][j,4]
                naca = int(sections[i][j,5])
                f.write(f'#-----------------------------------------\n\
# Xle           Yle            Zle           chord           angle  \n\
SECTION\n\
{Xle:.4f}        {Yle:.4f}        {Zle:.4f}        {chord:.4f}        {twist:.4f}\n\
NACA\n{naca:04d}\n')
                if ctrl[i]:
                    if ctrl[i][j]:
                        f.write(f'#Cname    Cgain    Xhinge           HingeVec        SgnDup\n\
CONTROL\n\
{ctrl[i][j][0]}   {ctrl[i][j][1][0]:.4f}   {ctrl[i][j][1][1]:.4f}   {ctrl[i][j][1][2]:.4f}   {ctrl[i][j][1][3]:.4f}   {ctrl[i][j][1][4]:.4f}   {ctrl[i][j][1][5]:.4f}\n')
                    
        
#%%

def duplicateSections(sects_in,y_mirror):
    
    sects_out = []
    
    for i in range(len(sects_in)):
        if y_mirror[i][0]:
            sects_out += [torch.concat([sects_in[i][1:,:].flipud()*torch.tensor([[1,-1,1,1,1, 1]]),sects_in[i]],dim=0).clone()]
    return sects_out

#%%
def makeMassFile(filename,mass_tot,CoM_tot,I_tot):

    with open(filename+'.mass', 'w') as f:
        
        x = CoM_tot[0].cpu().numpy()
        y = CoM_tot[1].cpu().numpy()
        z = CoM_tot[2].cpu().numpy()
        Ixx = I_tot[0,0].cpu().numpy()
        Iyy = I_tot[1,1].cpu().numpy()
        Izz = I_tot[2,2].cpu().numpy()
        Ixy = I_tot[0,1].cpu().numpy()
        Ixz = I_tot[0,2].cpu().numpy()
        Iyz = I_tot[1,2].cpu().numpy()
        
        mass_tot_g = mass_tot*1000
    
        f.write(f"#-------------------------------------------------\n\
#  {filename} {mass_tot_g:.0f}g \n\
#\n\
#  Dimensional unit and parameter data.\n\
#  Mass & Inertia breakdown.\n\
#-------------------------------------------------\n\
\n\
#  Names and scalings for units to be used for trim and eigenmode calculations.\n\
#  The Lunit and Munit values scale the mass, xyz, and inertia table data below.\n\
#  Lunit value will also scale all lengths and areas in the AVL input file.\n\
\n\
Lunit = 1.0 m\n\
Munit = 1.0 kg\n\
Tunit = 1.0 s\n\
\n\
#------------------------- \n\
#  Gravity and density to be used as default values in trim setup (saves runtime typing).\n\
#  Must be in the unit names given above (m,kg,s).\n\
g   = 9.81\n\
rho = 1.225\n\
\n\
#-------------------------\n\
#  Mass & Inertia breakdown.\n\
#  x y z  is location of item's own CG.\n\
#  Ixx... are item's inertias about item's own CG.\n\
#\n\
#  x,y,z system here must be exactly the same one used in the .avl input file\n\
#     (same orientation, same origin location, same length units)\n\
#\n\
# mass       x        y       z      Ixx       Iyy      Izz       Ixy       Ixz       Iyz  \n\
{mass_tot:.5f}  {x:.5f}  {y:.5f}  {z:.5f}  {Ixx:.5f}  {Iyy:.5f}  {Izz:.5f}  {Ixy:.5f}  {Ixz:.5f}  {Iyz:.5f}")
        
#%%

def run_commands(executable_path, commands):
    
    # Start the executable
    process = subprocess.Popen(executable_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Send the commands
    for cmd in commands:
        process.stdin.write(cmd + '\n')
    # Wait for completion and fetch the output

    stdout, stderr = process.communicate()

    process.stdin.close() 
    return stdout, stderr

def run_AVL(filename,AlphaDeg,BetaDeg):

    exe_path = 'avl.exe'
        
    cmd_sequence = [
        'load',
        filename,
        'mass',
        filename+'.mass',
        'mset',
        '0',
        '',
        'oper',
        'a',
        'a',
        str(AlphaDeg),
        'b',
        'b',
        str(BetaDeg),
        'x',
        'o',
        't',
        '',
        'x']
    
    output, error = run_commands(exe_path, cmd_sequence)
    trimmed_out = output[-2620:-1600]
    print("Output:", trimmed_out)
    
    nums = []
    
    for i in range(len(trimmed_out)):
        char = trimmed_out[i]
        if char=='=':
            string = trimmed_out[i+3:i+11]

            if '\n\n  Sr' not in string:
                num = float(string)
                nums += [num]
          
    return nums[16], nums[19], nums[21], nums[24] , nums[25] 

def run_AVL2(filename,AlphaDeg,BetaDeg):

    exe_path = 'avl.exe'
        
    cmd_sequence = [
        'load',
        filename,
        'mass',
        filename+'.mass',
        'mset',
        '0',
        '',
        'oper',
        'a',
        'a',
        str(AlphaDeg),
        'b',
        'b',
        str(BetaDeg),
        'x',
        'st',
        filename+'.st']
    
    output, error = run_commands(exe_path, cmd_sequence)
    trimmed_out = output[-2620:-1600]
    print("Output:", trimmed_out)
          
    return 
#%%

def geo(naca_m,naca_p,nc,y_mirror,translate,sections,ns,ctrl_id):
    

    r = torch.zeros([1,0,1,3])
    r_a = torch.zeros([1,1,0,3])
    r_b = torch.zeros([1,1,0,3])
    
    sref = torch.tensor(0.0)
    gxj = []
    gyj = []
    n_hat = torch.zeros([0,3])

    grid_x = []
    grid_y = []
    grid_z = []
    cntrd_x_np = []
    cntrd_y_np = []
    cntrd_z_np = []
    vort_x_np = []
    vort_y_np = []
    vort_z_np = []
    n_hat_j_np = []
    n_hat_j_plo_np = []

    n = 0
    strips = 0

    for j in range(len(nc)):
        
        
        if y_mirror[j]:
            mirrored_section = sections[j][1:,:].flipud()
            mirrored_section[:,1] = -sections[j][1:,1]
            sections[j] = torch.cat([mirrored_section,sections[j]],dim=0)
            ns[j] = torch.cat([ns[j].flip(dims=[0]),ns[j]],dim=0)
            
            for k in range(len(ctrl_id[j])):
                ctrl_id_mirrored = ctrl_id[j][k][:]
                ctrl_id_mirrored[1] = 1 - ctrl_id_mirrored[1]
                ctrl_id[j] += [ctrl_id_mirrored]
            
        
        strips += torch.sum(ns[j])
        n+=torch.sum(nc[j]*ns[j])
        
        xj_ofs = sections[j][:,0]
        yj_ofs = sections[j][:,1]
        zj_ofs = sections[j][:,2]
        cj = sections[j][:,3]
        
        syj = yj_ofs.diff().abs()
        szj = zj_ofs.diff().abs()
        

        sxz_j = (syj**2 + szj**2).sqrt()
        
        
        sref += torch.sum(0.5*(cj[:-1] + cj[1:])*sxz_j)
        
        naca_x0 = torch.linspace(0,1,nc[j]+1)+(1/(nc[j]+1)/4)
        naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
        
        naca_term = 2*naca_m[j]*(naca_p[j] - naca_x)
        dy_dx = naca_term/naca_p[j]**2
        dyb_dx = naca_term/(naca_p[j]-1.)**2
        
        dy_dx[naca_x>naca_p[j]] = dyb_dx[naca_x>naca_p[j]]
        
        
        n_vec = torch.concat([-dy_dx.reshape([-1,1]),torch.zeros([nc[j],1]),torch.ones([nc[j],1])],dim=1)
        n_hat_j = n_vec/torch.sqrt(torch.sum(n_vec**2,dim=1)).reshape([-1,1])
        n_hat_j_plo = torch.zeros([nc[j],3])
        n_hat_j_plo[:,2]= 1.
        
        gxn,gyn = torch.meshgrid(torch.linspace(0,1,nc[j]+1),torch.linspace(0,1,ns[j][0]+1))
        gxj = gxn*torch.linspace(cj[0],cj[1],ns[j][0]+1).reshape(1,ns[j][0]+1) + torch.linspace(xj_ofs[0],xj_ofs[1],ns[j][0]+1).reshape(1,ns[j][0]+1)
        gyj = gyn*syj[0] 
        gzj = gyj*0 + torch.linspace(zj_ofs[0],zj_ofs[1],ns[j][0]+1).reshape(1,ns[j][0]+1)
        
        si = 0
        
        ang = torch.atan2(zj_ofs[1] - zj_ofs[0],syj[0])
        
        cosA =torch.cos(ang)
        sinA =torch.sin(ang)
        T_sx = torch.tensor([[1.,0.,0.],[0.,cosA,-sinA],[0.,sinA,cosA]])
        
        n_hat_ji = torch.matmul(T_sx, n_hat_j.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][0]])
        n_hat_ji_plo = torch.matmul(T_sx, n_hat_j_plo.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][0]])
        
        for i in range(syj.shape[0]-1):
            
            ang = torch.atan2(zj_ofs[i+2] - zj_ofs[i+1],syj[i+1])
            
            cosA =torch.cos(ang)
            sinA =torch.sin(ang)
            T_sx = torch.tensor([[1.,0.,0.],[0.,cosA,-sinA],[0.,sinA,cosA]])
            
            gxn,gyn = torch.meshgrid(torch.linspace(0,1,nc[j]+1),torch.linspace(0,1,ns[j][i+1]+1))
            gxi = gxn*torch.linspace(cj[i+1],cj[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1) + torch.linspace(xj_ofs[i+1],xj_ofs[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1)
            si += syj[i] 
            gyi = gyn*syj[i+1] + si
            gzi = gyi*0 + torch.linspace(zj_ofs[i+1],zj_ofs[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1)
        
            gxj = torch.concat([gxj,gxi[:,1:]],dim=1)
            gyj = torch.concat([gyj,gyi[:,1:]],dim=1)
            gzj = torch.concat([gzj,gzi[:,1:]],dim=1)

            n_hat_ji_plo = torch.concat([n_hat_ji_plo,torch.matmul(T_sx, n_hat_j_plo.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][i+1]])],dim=2)
            n_hat_ji = torch.concat([n_hat_ji,torch.matmul(T_sx, n_hat_j.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][i+1]])],dim=2)
            
        n_j = (gxj.shape[1]-1)*(nc[j])
        
        gxj += translate[j][0] 
        gyj += translate[j][1] 
        gzj += translate[j][2] 
        
        n_hat_ji_plo = n_hat_ji_plo.permute([1,2,0])
        n_hat_ji = n_hat_ji.permute([1,2,0])
        
        cntrd_x,cntrd_y,cntrd_z = grid2rc(gxj,gyj,gzj)

        grid_x += [gxj]
        grid_y += [gyj]
        grid_z += [gzj]
        
        cntrd_x_np += [cntrd_x.cpu().numpy()]
        cntrd_y_np += [cntrd_y.cpu().numpy()]
        cntrd_z_np += [cntrd_z.cpu().numpy()]
        
        n_hat_j_np += [n_hat_ji.cpu().numpy()]
        n_hat_j_plo_np += [n_hat_ji_plo.cpu().numpy()]
        
        r_j = torch.concat([cntrd_x.reshape([-1,1]),cntrd_y.reshape([-1,1]),cntrd_z.reshape([-1,1])],dim = 1).reshape(1,n_j,1,3)

        r_a_j = torch.concat([gxj[0:-1,0:-1].reshape([-1,1]) + 0.25*gxj[:,0:-1].diff(dim=0).reshape(-1,1),gyj[0:-1,0:-1].reshape([-1,1]),gzj[0:-1,0:-1].reshape([-1,1])],dim = 1).reshape(1,1,n_j,3)
        r_b_j = torch.concat([gxj[0:-1,1:].reshape([-1,1]) +  0.25*gxj[:,1:].diff(dim=0).reshape(-1,1),gyj[0:-1,1:].reshape([-1,1]),gzj[0:-1,1:].reshape([-1,1])],dim = 1).reshape(1,1,n_j,3)
        
        r = torch.concat([r,r_j],dim=1)
        r_a = torch.concat([r_a,r_a_j],dim=2)
        r_b = torch.concat([r_b,r_b_j],dim=2)
        
        n_hat = torch.concat([n_hat,n_hat_ji.reshape([-1,3])],dim=0)
        
        vort_x_np += [(0.5*(r_a_j[:,:,:,0] + r_b_j[:,:,:,0] ).reshape(nc[j], strips)).cpu().numpy()]
        vort_y_np += [(0.5*(r_a_j[:,:,:,1] + r_b_j[:,:,:,1] ).reshape(nc[j], strips)).cpu().numpy()]
        vort_z_np += [(0.5*(r_a_j[:,:,:,2] + r_b_j[:,:,:,2] ).reshape(nc[j], strips)).cpu().numpy()]
        
        
    vort_xyz = [vort_x_np,vort_y_np,vort_z_np]    
    grid_xyz = [grid_x,grid_y,grid_z]    
    cntrd_xyz = [cntrd_x_np,cntrd_y_np,cntrd_z_np]     
    n_hat = n_hat.unsqueeze(0).unsqueeze(0)
    ri = 0.5*(r_a + r_b).permute([0,2,1,3])
    li = (r_b - r_a)
    
    calc_vars = [n_hat, sref, n, r, ri, r_a, r_b, li]
    plot_vars = [vort_xyz, grid_xyz, cntrd_xyz, n_hat_j_np, strips, n_hat_j_plo_np]
    
    return calc_vars, plot_vars
#%%
def ctrlvars(m,n,ctrl_id,ns,nc,grid_xyz):
    
    
    
    n_component_old = 0

    delta_ctrl = torch.zeros([n,0])

    for component in range(len(ctrl_id)):
        
        n_strip_comp = torch.sum(ns[component])
        
        idx_mat = torch.linspace(0,(n_strip_comp)*(nc[component])-1,(n_strip_comp)*(nc[component])).reshape([nc[component],n_strip_comp])
        for surface in range(len(ctrl_id[component])):

            naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
            naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
            
            if nc[component]>1:
                xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
            else:
                xid = 0
            
            ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
            ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
            
            y_prcnt_a = ctrl_id[component][surface][1][0]
            y_prcnt_b = ctrl_id[component][surface][1][1]
            
            yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()
            idx_surf = idx_mat[xid:,yid].reshape([-1,1]) + n_component_old
            
            ctrl_hot = torch.zeros([n,1])
            ctrl_hot[idx_surf.int(),0] = 1

            delta_ctrl = torch.concat([delta_ctrl,ctrl_hot],dim=1)
            
        n_component_old += (n_strip_comp)*(nc[component])
        
    ctrl_vec = torch.zeros([1,delta_ctrl.shape[1]])
    # counter = 0
    for component in range(len(ctrl_id)):
        for surface in range(len(ctrl_id[component])):
            ctrl_vec[0,surface] = torch.tensor(ctrl_id[component][surface][2])*torch.pi/180.

    ctrl_mat = ctrl_vec*torch.ones([m,1,1])    
    
    return delta_ctrl, ctrl_mat
        
#%%

def sol(delta_ctrl,ctrl_mat,U_vel,n_hat,LU, pivots, Vi_ri, li, sref):
    ctrl_tns = delta_ctrl*ctrl_mat
    ctrl_RHS = ctrl_tns.sum(dim=2).permute([1, 0])
    RHS = torch.sum(U_vel*n_hat,dim = 3).permute([2,0,1]).squeeze() + ctrl_RHS
    Gamma = torch.linalg.lu_solve(LU, pivots, RHS)
    V = torch.matmul(Vi_ri,Gamma.unsqueeze(0)).permute(2,1,0).unsqueeze(1) - U_vel
    Fi = 2*torch.cross(V,li)*Gamma.permute([1,0]).unsqueeze(1).unsqueeze(3)/sref
    F = (Fi).sum(dim=2).permute([2,0,1]).squeeze()
    
    return F, Gamma, Fi
#%%
def velrot(AlphaDeg,BetaDeg,m,n):
    
    alpha = torch.tensor(AlphaDeg*torch.pi/180.)
    beta = torch.tensor(BetaDeg*torch.pi/180.)
    cosA =torch.cos(alpha)
    sinA =torch.sin(alpha)
    cosB =torch.cos(beta)
    sinB =torch.sin(beta)
    
    TX = torch.tensor([[1.,0.,0.],[0.,cosB,-sinB],[0.,sinB,cosB]])
    TY = torch.tensor([[cosA,0.,sinA],[0.,1.,0.],[-sinA,0.,cosA]])
    
    U_vel = torch.zeros([m,1,n,3])
    U_vel[:,:,:,0] = -torch.cos(alpha)*torch.cos(beta)
    U_vel[:,:,:,1] = torch.sin(beta)
    U_vel[:,:,:,2] = -torch.sin(alpha)*torch.cos(beta)
    
    return U_vel, torch.matmul(TY,TX)
    
#%%

def getpivots(Vi,n_hat,n):
    Vi_infl = torch.sum(Vi*n_hat.reshape([1,-1,1,3]),dim = 3).reshape(n,n)
    return torch.linalg.lu_factor(Vi_infl)

def Fout(C_DYL,F,strips,n,nc,m,dt,sref,AlphaDeg,BetaDeg):
    C_DYL_np = C_DYL.cpu().numpy()


    print('------------------------------------------------------------------')
    print('# Surfaces =',len(nc))
    print('# Strips   =',strips.cpu().numpy())
    print('# Vortices =',n.cpu().numpy())
    print('\nSref       =',sref.cpu().numpy())
    print('\nAlpha      =',AlphaDeg)
    print('Beta       =',BetaDeg)
    print('\nCXtot      =',-F[0,0].cpu().numpy())
    print('CYtot      =',F[1,0].cpu().numpy())
    print('CZtot      =',-F[2,0].cpu().numpy())
    print('\nCLtot      =',C_DYL_np[2,0])
    print('CDind      =',C_DYL_np[0,0])
    print('\nSPS        =',m/dt)
    print('------------------------------------------------------------------')

#%%

def geoplot(ns,nc,ctrl_id,grid_xyz,cntrd_xyz,n_hat_j_np):

    plt.close('all')
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for i in range(len(ns)):
        ax.plot_wireframe(grid_xyz[0][i].cpu().numpy(),
                          grid_xyz[1][i].cpu().numpy(),
                          grid_xyz[2][i].cpu().numpy(),color='white', linewidth = 0.5)
        ax.quiver(cntrd_xyz[0][i].reshape(-1,1),
                  cntrd_xyz[1][i].reshape(-1,1),
                  cntrd_xyz[2][i].reshape(-1,1),
                  n_hat_j_np[i][:,:,0].reshape([-1,1]),
                  n_hat_j_np[i][:,:,1].reshape([-1,1]),
                  n_hat_j_np[i][:,:,2].reshape([-1,1]),
                  length=0.1, normalize=True, color = 'blue', linewidth = 1)
    
    for component in range(len(ctrl_id)):
        for surface in range(len(ctrl_id[component])):
            naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
            naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
            if nc[component]>1:
                xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
            else:
                xid = 0
            ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
            
            ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
            
            y_prcnt_a = ctrl_id[component][surface][1][0]
            y_prcnt_b = ctrl_id[component][surface][1][1]
            
            yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()
    
            if yid.shape[0]>0:
            
                yid_a = torch.min(yid)
                yid_b = torch.max(yid)+1
         
                ax.plot_surface(grid_xyz[0][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                            grid_xyz[1][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                            grid_xyz[2][component][xid:,yid_a:yid_b+1].cpu().numpy(),color=[1.,0.,1.], linewidth = 1)
    
    ax.set_aspect('equal')
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=225)
    # Transparent spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    
#%%

def solplot(ns,nc,ctrl_id,vort_xyz,grid_xyz,cntrd_xyz,n_hat_j_plo_np,Gamma,plot_id,plot_scale):

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax = fig.add_subplot(2, 1, 2, projection='3d')
    
    idx_a = 0
    
    
    for component in range(len(ns)):
    
        idx_b = idx_a + (torch.sum(ns[component]))*(nc[component])
        
        Gamma0_plo = (Gamma[idx_a:idx_b,plot_id].reshape([nc[component],-1])*plot_scale).cpu().numpy()
    
        idx_a = idx_b+0
        ax.plot_wireframe(grid_xyz[0][component].cpu().numpy(),
                          grid_xyz[1][component].cpu().numpy(),
                          grid_xyz[2][component].cpu().numpy(),color=[1.,0.,1.],linewidth=0.5)
        
        for strip in range((grid_xyz[0][component].shape[1]-1)):
            
            ax.plot(cntrd_xyz[0][component][:,strip],
                    cntrd_xyz[1][component][:,strip],
                    cntrd_xyz[2][component][:,strip],'y+')
            
            ax.plot_wireframe(vort_xyz[0][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,0],
                              vort_xyz[1][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,1],
                              vort_xyz[2][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,2],
                              color=[0.,1.,0.],linewidth=0.5)
            
            ax.plot_wireframe(vort_xyz[0][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,0],
                              vort_xyz[1][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,1],
                              vort_xyz[2][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,2],
                              color=[1.,0.,0.],linewidth=0.5)
            
            
    
    
    for component in range(len(ctrl_id)):
        for surface in range(len(ctrl_id[component])):
            naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
            naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
            
            if nc[component]>1:
                xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
            else:
                xid = 0
            ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
            
            ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
            
            y_prcnt_a = ctrl_id[component][surface][1][0]
            y_prcnt_b = ctrl_id[component][surface][1][1]
            
            
            
            yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()
    
            if yid.shape[0]>0:
    
                yid_a = torch.min(yid)
                yid_b = torch.max(yid)+1
            
                ax.plot_wireframe(grid_xyz[0][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                              grid_xyz[1][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                              grid_xyz[2][component][xid:,yid_a:yid_b+1].cpu().numpy(),color=[1.,1.,1.], linewidth = 1)
    
    ax.set_aspect('equal')
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=225)
    # Transparent spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    
    return ax

#%%

def streamplot(plot_id,start_x,step_const,n_stream_y,n_stream_z,Gamma,AlphaDeg,BetaDeg,r_a,r_b,n,ax):
    
    gamma0 = Gamma[:,plot_id].reshape(1,1,-1,1)
    
    ry,rz = torch.meshgrid(4.*torch.linspace(-1.0,1.0,n_stream_y),4.*torch.linspace(-1.,1.,n_stream_z))
    
    start_xyz = torch.concat([start_x*torch.ones([n_stream_z*n_stream_y,1]),ry.reshape([-1,1]),rz.reshape([-1,1])],dim = 1).reshape(1,-1,1,3)
    
    nt = 100
    streams = torch.zeros(nt,1,n_stream_z*n_stream_y,1,3)
    streams[0,:] = start_xyz;
    
    U_vel_n = torch.zeros([1,n_stream_z*n_stream_y,3])
    U_vel_n[:,:,0] = torch.cos(torch.tensor(AlphaDeg*torch.pi/180.))*torch.cos(torch.tensor(BetaDeg*torch.pi/180.))
    U_vel_n[:,:,1] = torch.sin(torch.tensor(BetaDeg*torch.pi/180.))
    U_vel_n[:,:,2] = torch.sin(torch.tensor(AlphaDeg*torch.pi/180.))*torch.cos(torch.tensor(BetaDeg*torch.pi/180.))
    
    x_hat = torch.zeros(1,1,n,3)
    x_hat[0,0,:,0] = 1     
    
    for i in range(nt-1):
        
        Vi = VORVELC_VEC(streams[i,:],r_a,r_b, 0.0, torch.tensor(1.).bool())
        
        V = torch.sum(Vi*gamma0,dim=2) + U_vel_n
        
        streams[i+1,:] = streams[i,:] + V.reshape(1,-1,1,3)*step_const
    
    streams_cpu = streams.cpu().numpy()
    
    i = 0
    
    for i in range(n_stream_z*n_stream_y):
        
        stream_xi = streams_cpu[:,0,i,0,0]
        stream_yi = streams_cpu[:,0,i,0,1]
        stream_zi = streams_cpu[:,0,i,0,2]
        
        ax.plot3D(stream_xi,stream_yi,stream_zi,'c')
        
    ax.set_aspect('equal')
    
#%%

def VORVEL(X,Y,Z,LBOUND,X1,Y1,Z1,X2,Y2,Z2,BETA):

    PI4INV  = torch.tensor(0.079577472)
    A = torch.zeros(3)
    B = torch.zeros(3)
    AXB = torch.zeros(3)
    A[0] = (X1 - X)/BETA
    A[1] =  Y1 - Y
    A[2] =  Z1 - Z
    
    B[0] = (X2 - X)/BETA
    B[1] =  Y2 - Y
    B[2] =  Z2 - Z
    
    ASQ = A[0]**2 + A[1]**2 + A[2]**2
    BSQ = B[0]**2 + B[1]**2 + B[2]**2
    
    AMAG = ASQ.sqrt()
    BMAG = BSQ.sqrt()
    
  
    
    AXB[0] = A[1]*B[2] - A[2]*B[1]
    AXB[1] = A[2]*B[0] - A[0]*B[2]
    AXB[2] = A[0]*B[1] - A[1]*B[0]
 
    ADB = A[0]*B[0] + A[1]*B[1] + A[2]*B[2]
 
    DEN = AMAG*BMAG + ADB
 
    T = LBOUND*(1.0/AMAG + 1.0/BMAG) / DEN
    
    AXISQ = A[2]**2 + A[1]**2
 
    ADI = A[0]
    RSQ = AXISQ
 
    TA = - (1.0 - ADI/AMAG) / RSQ
 

 
    BXISQ = B[2]**2 + B[1]**2
 
    BDI = B[0]
    RSQ = BXISQ
 
    TB =   (1.0 - BDI/BMAG) / RSQ
 
    U = PI4INV*(AXB[0]*T)/ BETA
    V = PI4INV*(AXB[1]*T + A[2]*TA + B[2]*TB)
    W = PI4INV*(AXB[2]*T - A[1]*TA - B[1]*TB)

    return  U, V, W
#%%
def VORVELC(X,Y,Z,LBOUND,X1,Y1,Z1,X2,Y2,Z2,BETA, RCORE):

    PI4INV  = torch.tensor(0.079577472)
    A = torch.zeros(3)
    B = torch.zeros(3)
    AXB = torch.zeros(3)
    A[0] = (X1 - X)/BETA
    A[1] =  Y1 - Y
    A[2] =  Z1 - Z
    
    B[0] = (X2 - X)/BETA
    B[1] =  Y2 - Y
    B[2] =  Z2 - Z
    
    ASQ = A[0]**2 + A[1]**2 + A[2]**2
    BSQ = B[0]**2 + B[1]**2 + B[2]**2
    
    AMAG = ASQ.sqrt()
    BMAG = BSQ.sqrt()
    
    
    AXB[0] = A[1]*B[2] - A[2]*B[1]
    AXB[1] = A[2]*B[0] - A[0]*B[2]
    AXB[2] = A[0]*B[1] - A[1]*B[0]
    AXBSQ = AXB[0]**2 + AXB[1]**2 + AXB[2]**2
    
    ADB = A[0]*B[0] + A[1]*B[1] + A[2]*B[2]
    ALSQ = ASQ + BSQ - 2.0*ADB

    T = LBOUND*((BSQ-ADB)/torch.sqrt(BSQ+RCORE**2) + (ASQ-ADB)/torch.sqrt(ASQ+RCORE**2) ) / (AXBSQ + ALSQ*RCORE**2)

    U = AXB[0]*T
    V = AXB[1]*T
    W = AXB[2]*T
  
    AXISQ = A[2]**2 + A[1]**2
  
    ADI = A[0]
    RSQ = AXISQ
  
    TA = - (1.0 - ADI/AMAG) / (RSQ + RCORE**2)
  
    V = V + A[2]*T
    W = W - A[1]*T
  
    BXISQ = B[2]**2 + B[1]**2
  
    BDI = B[0]
    RSQ = BXISQ
  
    TB =   (1.0 - BDI/BMAG) / (RSQ + RCORE**2)
  
    U = PI4INV*(AXB[0]*T)/ BETA 
    V = PI4INV*(AXB[1]*T + A[2]*TA + B[2]*TB)
    W = PI4INV*(AXB[2]*T - A[1]*TA - B[1]*TB)
    
    

    return  U, V, W

#%%
def grid2rc(gxj,gyj,gzj):
    
    rc_x = 0.125*(gxj[0:-1,0:-1] + gxj[0:-1,1:]) + 0.375*(gxj[1:,0:-1] + gxj[1:,1:])
    rc_y = 0.25*(gyj[0:-1,0:-1] + gyj[1:,0:-1] + gyj[0:-1,1:] + gyj[1:,1:]) 
    rc_z = 0.25*(gzj[0:-1,0:-1] + gzj[1:,0:-1] + gzj[0:-1,1:] + gzj[1:,1:]) 

    return rc_x, rc_y, rc_z

#%%

def linmoi(xy,z):
    
    m = (xy[1,1]-xy[0,1])/(xy[1,0]-xy[0,0])
    b = xy[1,1] - m*xy[1,0]

    Ixx = -2*(xy[0,0]-xy[1,0])*(2*b + m*xy[0,0] + m*xy[1,0])*(2*b**2 + 2*b*m*xy[0,0] + 2*b*m*xy[1,0] + m**2*xy[0,0]**2 + m**2*xy[1,0]**2 + 6*z**2)/24
    Iyy = (6*m*xy[1,0]**4 - 8*b*xy[0,0]**3 - 12*m*xy[0,0]**2*z**2 - 24*b*xy[0,0]*z**2-6*m*xy[0,0]**4 + 8*b*xy[1,0]**3 + 12*m*xy[1,0]**2*z**2 + 24*b*xy[1,0]*z**2)/24
    Izz = (8*b**3*xy[1,0] - 8*b**3*xy[0,0] - 12*b**2*m*xy[0,0]**2 + 12*b**2*m*xy[1,0]**2 - 8*b*m**2*xy[0,0]**3 + 8*b*m**2*xy[1,0]**3 - 8*b*xy[0,0]**3 + 8*b*xy[1,0]**3 - 2*m**3*xy[0,0]**4 + 2*m**3*xy[1,0]**4 - 6*m*xy[0,0]**4 + 6*m*xy[1,0]**4)/24
    Ixy = (6*b**2*xy[0,0]**2 - 6*b**2*xy[1,0]**2 + 8*b*m*xy[0,0]**3 - 8*b*m*xy[1,0]**3 + 3*m**2*xy[0,0]**4 - 3*m**2*xy[1,0]**4)/24
    Ixz = (8*m*z*xy[0,0]**3 + 12*b*z*xy[0,0]**2 - 8*m*z*xy[1,0]**3 - 12*b*z*xy[1,0]**2)/24
    Iyz = 4*z*(xy[0,0]-xy[1,0])*(3*b**2 + 3*b*m*xy[0,0] + 3*b*m*xy[1,0]+m**2*xy[0,0]**2 + m**2*xy[0,0]*xy[1,0] + m**2*xy[1,0]**2)/24
    
    I = torch.tensor([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])

    return torch.where(I.isnan(),0,I)
 
#%%   

def linmois(xyp,zp):
    
    m = (xyp[:,1,1]-xyp[:,0,1])/(xyp[:,1,0]-xyp[:,0,0])
    b = xyp[:,1,1] - m*xyp[:,1,0]

    Ixx = -2*(xyp[:,0,0]-xyp[:,1,0])*(2*b + m*xyp[:,0,0] + m*xyp[:,1,0])*(2*b**2 + 2*b*m*xyp[:,0,0] + 2*b*m*xyp[:,1,0] + m**2*xyp[:,0,0]**2 + m**2*xyp[:,1,0]**2 + 6*zp**2)/24
    Iyy = (6*m*xyp[:,1,0]**4 - 8*b*xyp[:,0,0]**3 - 12*m*xyp[:,0,0]**2*zp**2 - 24*b*xyp[:,0,0]*zp**2-6*m*xyp[:,0,0]**4 + 8*b*xyp[:,1,0]**3 + 12*m*xyp[:,1,0]**2*zp**2 + 24*b*xyp[:,1,0]*zp**2)/24
    Izz = (8*b**3*xyp[:,1,0] - 8*b**3*xyp[:,0,0] - 12*b**2*m*xyp[:,0,0]**2 + 12*b**2*m*xyp[:,1,0]**2 - 8*b*m**2*xyp[:,0,0]**3 + 8*b*m**2*xyp[:,1,0]**3 - 8*b*xyp[:,0,0]**3 + 8*b*xyp[:,1,0]**3 - 2*m**3*xyp[:,0,0]**4 + 2*m**3*xyp[:,1,0]**4 - 6*m*xyp[:,0,0]**4 + 6*m*xyp[:,1,0]**4)/24
    Ixy = (6*b**2*xyp[:,0,0]**2 - 6*b**2*xyp[:,1,0]**2 + 8*b*m*xyp[:,0,0]**3 - 8*b*m*xyp[:,1,0]**3 + 3*m**2*xyp[:,0,0]**4 - 3*m**2*xyp[:,1,0]**4)/24
    Ixz = (8*m*zp*xyp[:,0,0]**3 + 12*b*zp*xyp[:,0,0]**2 - 8*m*zp*xyp[:,1,0]**3 - 12*b*zp*xyp[:,1,0]**2)/24
    Iyz = 4*zp*(xyp[:,0,0]-xyp[:,1,0])*(3*b**2 + 3*b*m*xyp[:,0,0] + 3*b*m*xyp[:,1,0]+m**2*xyp[:,0,0]**2 + m**2*xyp[:,0,0]*xyp[:,1,0] + m**2*xyp[:,1,0]**2)/24

    Ix = torch.cat([Ixx.unsqueeze(dim=1),Ixy.unsqueeze(dim=1),Ixz.unsqueeze(dim=1)],dim=1).unsqueeze(dim=1)
    Iy = torch.cat([Ixy.unsqueeze(dim=1),Iyy.unsqueeze(dim=1),Iyz.unsqueeze(dim=1)],dim=1).unsqueeze(dim=1)
    Iz = torch.cat([Ixz.unsqueeze(dim=1),Iyz.unsqueeze(dim=1),Izz.unsqueeze(dim=1)],dim=1).unsqueeze(dim=1)
    I = torch.cat([Ix,Iy,Iz],dim=1)
    
    return torch.where(I.isnan(),0,I)



def triareas(xyz):
    
    vec = torch.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    vec_norm = (vec**2).sum(dim=1).sqrt()
    area = 0.5*vec_norm.unsqueeze(dim=1)
    
    CoM = xyz.mean(dim=1)
    
    return area, CoM

def quadareas(xyz):
    
    t0 = xyz.roll(dims=1,shifts=0)[:,:3,:]
    t1 = xyz.roll(dims=1,shifts=2)[:,:3,:]

    at0, CoMt0 = triareas(t0)
    at1, CoMt1 = triareas(t1)

    CoMq = (CoMt0*at0 + CoMt1*at1)/(at0+at1)

    areaq = at0 + at1
    
    return areaq, CoMq

#%%  

def triarea(xyz):
    
    vec = torch.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
    vec_norm = (vec**2).sum().sqrt()
    area = 0.5*vec_norm
    
    CoM = xyz.mean(dim=0)
    
    return area, CoM


#%%

def quadarea(xyz):
    
    # xyz in cw or ccw order
    
    xyz = xyz.unsqueeze(dim=1)

    t0 = xyz.roll(dims=0,shifts=0)[:3,:]
    t1 = xyz.roll(dims=0,shifts=2)[:3,:]

    at0, CoMt0 = triarea(t0)
    at1, CoMt1 = triarea(t1)

    CoMq = (CoMt0*at0 + CoMt1*at1)/(at0+at1)
    
    areaq = at0 + at1
    
    return areaq, CoMq

#%%

def trimoi(xyz):
    xyz = xyz[xyz[:,0].sort(dim=0)[1]]
    # xyz = torch.sort(xyz,dim=0)[0]
    
    vec = torch.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
    vec_norm = (vec**2).sum().sqrt()
    norm_vec = vec/vec_norm
    
    ac = xyz[2]-xyz[0]
    ac_norm = (ac**2).sum().sqrt()
    norm_ac = ac/ac_norm

    vec3 = torch.cross(norm_ac,norm_vec)

    rot = torch.concat([norm_ac.unsqueeze(1),vec3.unsqueeze(1),norm_vec.unsqueeze(1)],dim=1)

    xyzp = torch.matmul(xyz,rot)
    
    xyp = xyzp[:,0:2].unsqueeze(1)
    zp = xyzp[0,2]

    Iab = linmoi(torch.cat([xyp[0],xyp[1]],dim=0),zp)
    Ibc = linmoi(torch.cat([xyp[1],xyp[2]],dim=0),zp)
    Iac = linmoi(torch.cat([xyp[0],xyp[2]],dim=0),zp)

    m = (xyzp[2,1]-xyzp[0,1])/(xyzp[2,0]-xyzp[0,0])
    b = xyzp[2,1] - m*xyzp[2,0]

    mode = (xyzp[1,1] > m*xyzp[1,0]+b)*2 - 1

    Ip = (Iab - Iac  + Ibc)*mode

    I = torch.matmul(torch.matmul(rot,Ip),rot.permute([1,0]))
    
    return I

def trimois(xyz):
    
    vec = torch.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    vec_norm = (vec**2).sum(dim=1).sqrt()
    norm_vec = vec/vec_norm.unsqueeze(dim=1)
    
    ac = xyz[:,2]-xyz[:,0]
    ac_norm = (ac**2).sum(dim=1).sqrt()
    norm_ac = ac/ac_norm.unsqueeze(dim=1)
    
    vec3 = torch.cross(norm_ac,norm_vec)
    
    rot = torch.concat([norm_ac.unsqueeze(1),vec3.unsqueeze(1),norm_vec.unsqueeze(1)],dim=1).permute(0,2,1)
    
    xyzp = torch.matmul(xyz,rot)
    
    xyp = xyzp[:,:,0:2]
    zp = xyzp[:,:,2].unsqueeze(dim=2)[:,0].squeeze()
    
    
    L1 = torch.cat([xyp[:,0].unsqueeze(dim=1),xyp[:,1].unsqueeze(dim=1)],dim=1)
    L2 = torch.cat([xyp[:,1].unsqueeze(dim=1),xyp[:,2].unsqueeze(dim=1)],dim=1)
    L3 = torch.cat([xyp[:,0].unsqueeze(dim=1),xyp[:,2].unsqueeze(dim=1)],dim=1)
    
    Iab = linmois(L1,zp)
    Ibc = linmois(L2,zp)
    Iac = linmois(L3,zp)
    
    
    m = (xyzp[:,2,1]-xyzp[:,0,1])/(xyzp[:,2,0]-xyzp[:,0,0])
    b = xyzp[:,2,1] - m*xyzp[:,2,0]
    
    
    mode = (xyzp[:,1,1] > m*xyzp[:,1,0]+b)*2 - 1
    
    Ip = (Iab - Iac  + Ibc)*mode.unsqueeze(dim=1).unsqueeze(dim=2)
    
    I = torch.matmul(torch.matmul(rot,Ip),rot.permute([0,2,1])).sum(dim=0)
    
    return I

#%%

def quadmoi(xyz):
    
    # xyz in cw or ccw order
    
    xyz = xyz.unsqueeze(dim=1)

    t1 = torch.cat([xyz[0],
                    xyz[1],
                    xyz[2]],dim=0)
    
    t2 = torch.cat([xyz[0],
                    xyz[2],
                    xyz[3]],dim=0)
    
    I1 = trimoi(t1)
    I2 = trimoi(t2)
    
    I = I1 + I2
    
    return I

def quadmois(xyz):
    
    t1 = torch.cat([xyz[:,0].unsqueeze(dim=0),
                    xyz[:,1].unsqueeze(dim=0),
                    xyz[:,2].unsqueeze(dim=0)],dim=0)

    t2 = torch.cat([xyz[:,0].unsqueeze(dim=0),
                    xyz[:,2].unsqueeze(dim=0),
                    xyz[:,3].unsqueeze(dim=0)],dim=0)


    idx1 = t1[:,:,0].sort(dim=0)[1]
    t1_xsorted = t1.permute(2,0,1).gather(1,idx1.repeat([3,1,1])).permute(2,1,0)
    idx2 = t2[:,:,0].sort(dim=0)[1]
    t2_xsorted = t2.permute(2,0,1).gather(1,idx2.repeat([3,1,1])).permute(2,1,0)

    
    I1 = trimois(t1_xsorted)
    I2 = trimois(t2_xsorted)
    
    I = I1 + I2
    
    return I

def pointmoi(xyz,mass):

    ixxn = (mass.squeeze()*(xyz[:,2]**2 + xyz[:,1]**2)).sum()
    ixyn = (mass.squeeze()*(-xyz[:,0]*xyz[:,1])).sum()
    ixzn = (mass.squeeze()*(-xyz[:,0]*xyz[:,2])).sum()
    iyyn = (mass.squeeze()*(xyz[:,0]**2 + xyz[:,2]**2)).sum()
    iyzn =(mass.squeeze()*(-xyz[:,2]*xyz[:,1])).sum()
    izzn = (mass.squeeze()*(xyz[:,0]**2 + xyz[:,1]**2)).sum()

    return torch.tensor([[ixxn,ixyn,ixzn],[ixyn,iyyn,iyzn],[ixzn,iyzn,izzn]])

#%%

def translateSections(sections,translate):
    for i in range(len(sections)):
        
        sections[i][:,:3] += translate[i].unsqueeze(0)

    return sections

def sections2quads(sections):

    xyz = []    

    for i in range(len(sections)):
        
        xyz += [torch.cat([sections[i][:-1,:3].unsqueeze(dim=0),
               (sections[i][:-1,:3] + sections[i][:-1,3].unsqueeze(dim=1)*torch.tensor([[1., 0., 0.]])).unsqueeze(dim=0),
               (sections[i][1:,:3]  + sections[i][1:,3].unsqueeze(dim=1)*torch.tensor([[1., 0., 0.]])).unsqueeze(dim=0),
               sections[i][1:,:3].unsqueeze(dim=0)],dim=0).permute([1,0,2])]

    return torch.cat(xyz,dim=0)