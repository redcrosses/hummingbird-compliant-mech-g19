import yaml
import numpy as np
from ansys.mapdl.core import launch_mapdl


def vec(length, angle_deg, z = 0):
    rad = np.radians(angle_deg)
    return np.array([length * np.cos(rad), length * np.sin(rad), z])

def flexure_offsets(flex_len, vec1, vec2):
    norm_vec1 = vec1 / np.linalg.norm(vec1)

    ret1 = -norm_vec1*flex_len/2
    ret2 = norm_vec1*flex_len/2
    ret3 = np.dot(ret2, ret2) / np.dot(vec2, ret2) * vec2
    #The alternative point that can be used if the angle between vectors is acute
    ret4 = 2*ret2-ret3

    return (ret1, ret2, ret3, ret4)

mapdl = launch_mapdl(log_apdl="alternative/mechanism.inp")

with open("alternative/params.yaml", "r") as f:
    data = yaml.safe_load(f)

scale = data['prbm']['scale']
lengths = {k: v * scale for k, v in data['prbm']['lengths'].items()}
angles = data['prbm']['fabrication_angles']

vecs = {}
for i in range(1,10):
    vecs[f'r{i}'] = vec(lengths[f'L{i}'], angles[f'th{i}'])

fem_params = data['fem']
fem_dims = data['fem']['dimensions']
fem_flexures = data['fem']['dimensions']['flexures']

#---ELEMENT DEFINITION AND MATERIAL PROPERTIES---
mapdl.prep7()
element_str = f"""
ET, 1, BEAM188
KEYOPT,1,3,2
MP,EX,1,{fem_params['E_link']}     !Type 1 properties
MP,PRXY,1,0.35      !TODO outside def

MP,EX,2,{fem_params['E_flex']}     !Type 2 properties
MP,PRXY,2,0.3       !TODO outside def
"""
mapdl.input_strings(element_str)

#--CROSS-SECTIONS
mapdl.sectype(1, "BEAM", "RECT", "LINK")    #type: ignore
mapdl.secdata(fem_dims["link_thickness"], fem_dims["link_depth"])

mapdl.sectype(2, "BEAM", "RECT", "FLEXURE") #type: ignore
mapdl.secdata(fem_dims["flex_thickness"], fem_dims["flex_depth"])

#---KEYPOINTS---
geom = {}
geom['O2'] = np.array([0.0, 0.0, 0.0])
geom['O4'] = geom['O2'] + vecs['r1']
geom['A']  = geom['O2'] + vecs['r2']
geom['B']  = geom['A']  + vecs['r3']
geom['D']  = geom['O4'] + vecs['r7']
geom['C']  = geom['D']  - vecs['r8']
geom['P']  = geom['C']  + vecs['r9']

level_offset = np.array([0,0, fem_dims['z_offset']+fem_dims['link_depth']])

vecs["r10"] = geom["D"] - geom["B"]

kps = {}
kps[1] = geom['O2'] + level_offset
kps[2] = geom['A'] + level_offset
kps[3] = geom['A']

B_flexure_offsets = flexure_offsets(fem_flexures['B'], vecs['r3'], vecs['r10'])
kps[4] = geom['B'] + B_flexure_offsets[0]
kps[5] = geom['B'] + B_flexure_offsets[1]
kps[6] = geom['B'] + B_flexure_offsets[2]

D_flexure_offsets = flexure_offsets(fem_flexures['D'], vecs['r8'], vecs['r10'])
kps[7] = geom['D'] + D_flexure_offsets[0]
kps[8] = geom['D'] + D_flexure_offsets[1]
kps[9] = geom['D'] + D_flexure_offsets[2]

C_flexure_offsets = flexure_offsets(fem_flexures['C'], -vecs['r9'], -vecs['r8'])
kps[10] = geom['C'] + C_flexure_offsets[0]
kps[11] = geom['C'] + C_flexure_offsets[1]
kps[12] = geom['C'] + C_flexure_offsets[3]

P_flexure_offsets = flexure_offsets(fem_flexures['P'], vecs['r9'], vecs['r5'])
kps[13] = geom['P'] + P_flexure_offsets[0]
kps[14] = geom['P'] + P_flexure_offsets[1]
kps[15] = geom['P'] + P_flexure_offsets[2]

O4_flexure_offsets = flexure_offsets(fem_flexures['O4'], geom['O4'] - kps[9], np.array([0,1,0]))
kps[16] = geom['O4'] + O4_flexure_offsets[0]
kps[17] = geom['O4'] + O4_flexure_offsets[1]

for kp_id, (x, y, z) in kps.items():
     mapdl.k(kp_id, x, y, z)

#---LINES---
flexure_lines = [
    (4,5), #flexure
    (8,7), #flexure
    (11,10), #flexure
    (13,14), #flexure
    (16,17)  # ground O4
]

link_lines = [
    (1, 2),
    (3,4),
    (5,6),
    (6,9),
    (9,8),
    (7,12),
    (12,11),
    (10,13),
    (14,15),
    (15,3),
    (9,16)
]

# TODO export parameter configuration

#---CONFIGURE MESH---
flexure_ids = []
for p1, p2 in flexure_lines:
    line_id = mapdl.lstr(p1,p2)   #type: ignore

    mapdl.lsel("S", "LINE", "", line_id)    #type: ignore
    mapdl.latt(mat=2, type_=1, secnum=2)    #type: ignore
    mapdl.lesize(line_id, ndiv=15)        #type: ignore

link_ids = []
for p1, p2 in link_lines:
    line_id = mapdl.lstr(p1, p2)  #type: ignore

    mapdl.lsel("S", "LINE", "", line_id)    #type: ignore
    mapdl.latt(mat=1, type_=1, secnum=1)    #type: ignore
    mapdl.lesize(line_id, ndiv=5)        #type: ignore

#---MESH---
mapdl.allsel()
mapdl.lmesh("ALL")

#---PLOT MESH---
mapdl.eshape(1) #type: ignore
mapdl.open_gui()

mapdl.exit()