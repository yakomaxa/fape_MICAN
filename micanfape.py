from pymol import cmd
import numpy as np
import pymican
import torch
from einops import rearrange, repeat

# below: taken from pdbbasic
def coord2rotation(coord, eps=1e-8):
    x1 = coord[:,2] - coord[:,1]
    x1 = x1 / (repeat(torch.linalg.norm(x1, dim=1), 'l -> l c', c=3)+eps)
    x2 = coord[:,0] - coord[:,1]
    x2 = x2 - torch.einsum('l c, l -> l c', x1, (torch.einsum('l c, l c -> l', x1, x2)))
    x2 = x2 / (repeat(torch.linalg.norm(x2, dim=1), 'l -> l c', c=3)+eps)
    x3 = torch.cross(x1, x2)
    x3 = x3 / (repeat(torch.linalg.norm(x3, dim=1), 'l -> l c', c=3)+eps)
    return torch.stack([x1, x2, x3], dim=1).transpose(1,2)

def coord2translation(coord):
    return coord[:,1]

def coord_to_frame(coord: torch.Tensor) -> tuple:
    org_type = type(coord)
    coord = torch.tensor(coord) if org_type == np.ndarray else coord
    rot = coord2rotation(coord)
    trans = coord2translation(coord)
    trans, rot = (trans.cpu().numpy(), rot.cpu().numpy()) if org_type == np.ndarray else (trans, rot)
    return trans, rot

def frame_aligned_matrix(frame):
    trans, rot = frame
    trans = trans.unsqueeze(0) if len(trans.shape)==2 else trans
    rot = rot.unsqueeze(0) if len(rot.shape)==3 else rot
    m = n = trans.shape[1]
    co_mat = repeat(trans, 'b n c -> b m n c', m=m) - repeat(trans, 'b m c -> b m n c', n=n)
    rot_inv = rearrange(rot, 'b m c1 c2 -> b m c2 c1')
    rot_inv_mat = repeat(rot_inv, 'b m r c -> b m n r c', n=n)
    return torch.einsum('b m n r c, b m n c -> b m n r', rot_inv_mat, co_mat)

def FAPE(frame1, frame2, D=10, eps=1e-8, Z=10, mean=True):
    org_type1, org_type2 = type(frame1[0]), type(frame2[0])
    frame1 = (torch.tensor(v) for v in frame1) if org_type1 == np.ndarray else frame1
    frame2 = (torch.tensor(v) for v in frame2) if org_type2 == np.ndarray else frame2
    co1_mat = frame_aligned_matrix(frame1)
    co2_mat = frame_aligned_matrix(frame2)
    device = co1_mat.device
    fape_abs = torch.sqrt(torch.pow(co1_mat - co2_mat, 2).sum(dim=-1)+eps)
    fape_clamp = torch.min(fape_abs, torch.ones_like(fape_abs).to(device)*D)
    fape = fape_clamp.mean()/Z if mean else fape_clamp.mean(dim=[1,2])/Z
    fape = fape.cpu().numpy() if org_type1 == np.ndarray else fape
    return fape
## above: taken from pdb basic

#from pdbbasic import frame

def fape(target1,target2,mode="super"):
    name=cmd.get_unused_name()
    if (mode=="super"):
        result=cmd.super(target1, target2, object=name)
    elif (mode=="align"):
        result=cmd.align(target1, target2, object=name)

    raw_aln=cmd.get_raw_alignment(name)
    idx2resi = {}
    cmd.iterate(name, 'idx2resi[model, index] = resi', space={'idx2resi': idx2resi})
    # print residue pairs (residue number)
    indexes1=list()
    indexes2=list()
    for idx1, idx2 in raw_aln:
        indexes1.append(int(idx2resi[idx2])-1)
        indexes2.append(int(idx2resi[idx1])-1)
    slice1=sorted(set(indexes1),key=indexes1.index)
    slice2=sorted(set(indexes2),key=indexes2.index)

    coord1=cmd.get_coords(target1 + " and name n+ca+c+o").reshape([-1,4,3])
    coord2=cmd.get_coords(target2 + " and name n+ca+c+o").reshape([-1,4,3])

    frame1=coord_to_frame(coord1[slice1])
    frame2=coord_to_frame(coord2[slice2])
    cmd.delete(name)
    fape=FAPE(frame1,frame2)
    rmsd=result[0]
    print('RMSD= %f FAPE= %f' % (rmsd,fape))

def fape_super(target1,target2):
    fape(target1,target2,"super")

def fape_align(target1, target2):
    fape(target1, target2, "align")

cmd.extend("fsuper", fape_super)
cmd.auto_arg[0]['fsuper'] = cmd.auto_arg[0]['super']
cmd.auto_arg[1]['fsuper'] = cmd.auto_arg[1]['super']
cmd.extend("falign", fape_align)
cmd.auto_arg[0]['falign'] = cmd.auto_arg[0]['align']
cmd.auto_arg[1]['falign'] = cmd.auto_arg[1]['align']

def fape_mican(mobile, target, option="-s"):
    with tempfile.TemporaryDirectory() as dname:
        tmptarget = dname + "/target.pdb"
        tmpmobile = dname + "/mobile.pdb"
        cmd.save(tmptarget, target)
        cmd.save(tmpmobile, mobile)

        mican = pymican.mican()
        res = mican.align(tmpmobile, tmptarget,options=option)
        coords = cmd.get_coords(mobile)
        cmd.load_coords(res.translate_xyz(coords), mobile)
        slice1 = [int(n)-1 for n in list(res.alignment["residue2"])]
        slice2 = [int(n)-1 for n in list(res.alignment["residue1"])]

        coord1 = cmd.get_coords(target + " and name n+ca+c+o").reshape([-1, 4, 3])
        coord2 = cmd.get_coords(mobile + " and name n+ca+c+o").reshape([-1, 4, 3])

        frame1 = coord_to_frame(coord1[slice1])
        frame2 = coord_to_frame(coord2[slice2])
        fape = FAPE(frame1, frame2)
    print('RMSD= %f FAPE= %f TMscore= %f TMscore1= %f TMscore2= %f Dali= %f' %
          (res.rmsd, fape, res.TMscore,res.TMscore1,res.TMscore2, res.DALIscore))

cmd.extend("fmican", fape_mican)
cmd.auto_arg[0]['fmican'] = cmd.auto_arg[0]["align"]
cmd.auto_arg[1]['fmican'] = cmd.auto_arg[1]['align']
