import os
from pathlib import Path
import random
import time

import numpy as np

# from polnet.samplegeneration.synthetictomo.tomogram import Tomogram
# from polnet.tomofiles.mb_file import MbFile
# from polnet.utils import poly as pp
# from polnet.utils import lio

from polnet.tomogram import SynthTomo
from polnet.sample import MbFile
from polnet.utils import lio

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Common tomogram settings
# ROOT_PATH should point to the 'data' folder
ROOT_PATH = Path(__file__).resolve().parents[2] / "data"
NTOMOS = 1
VOI_SHAPE = (
    300,
    300,
    250,
)
VOI_OFFS = (
    4,
    4,
    4,
)
VOI_VSIZE = 10  # A/vx

# Lists with the features to simulate
MEMBRANES_LIST = [
    # "in_mbs/sphere.mbs",
    # "in_mbs/ellipse.mbs",
    # "in_mbs/toroid.mbs",
]

# HELIX_LIST = [
#     "in_helix/mt.hns",
#     "in_helix/actin.hns"
#     ]

PROTEINS_LIST = [
     "in_10A/4v4r_10A.pns"
    # "in_10A/3j9i_10A.pns",
    # "in_10A/5mrc_10A.pns",
    # "in_10A/4v7r_10A.pns",
    # "in_10A/2uv8_10A.pns",
    # "in_10A/4v94_10A.pns",
    # "in_10A/4cr2_10A.pns",
    # "in_10A/3qm1_10A.pns",
    # "in_10A/3h84_10A.pns",
    # "in_10A/3gl1_10A.pns",
    # "in_10A/3d2f_10A.pns",
    # "in_10A/3cf3_10A.pns",
    # "in_10A/2cg9_10A.pns",
    # "in_10A/1u6g_10A.pns",
    # "in_10A/1s3x_10A.pns",
    # "in_10A/1qvr_10A.pns",
    # "in_10A/1bxn_10A.pns",
 ]

MB_PROTEINS_LIST = [
    "in_10A/mb_6rd4_10A.pms",
#     "in_10A/mb_5wek_10A.pms",
#     "in_10A/mb_4pe5_10A.pms",
#     "in_10A/mb_5ide_10A.pms",
#     "in_10A/mb_5gjv_10A.pms",
#     "in_10A/mb_5kxi_10A.pms",
#     "in_10A/mb_5tj6_10A.pms",
#     "in_10A/mb_5tqq_10A.pms",
#     "in_10A/mb_5vai_10A.pms",
 ]

# Proportions list, specifies the proportion for each protein, this proportion is tried to be achieved but no guaranteed
# The total sum of this list must be 1
PROP_LIST = None  # [.4, .6]
if PROP_LIST is not None:
    assert sum(PROP_LIST) == 1

SURF_DEC = 0.9  # Target reduction factor for surface decimation (default None)

# Reconstruction tomograms
TILT_ANGS = np.arange(
    -60, 60, 3
)  # range(-90, 91, 3) # at MPI-B IMOD only works for ranges
DETECTOR_SNR = [1.0, 2.0]  # 0.2 # [.15, .25]
MALIGN_MN = 1
MALIGN_MX = 1.5
MALIGN_SG = 0.2

# OUTPUT FILES
# OUT_DIR = os.path.realpath(
#     ROOT_PATH + "/data_generated/development_all_features"
# )  # '/out_all_tomos_9-10' # '/only_actin' # '/out_rotations'
OUT_DIR = ROOT_PATH / "data_generated" / "output" / "output_sawlc"
os.makedirs(OUT_DIR, exist_ok=True)


TEM_DIR = OUT_DIR / "tem"
TOMOS_DIR = OUT_DIR / "tomos"
os.makedirs(TOMOS_DIR, exist_ok=True)
os.makedirs(TEM_DIR, exist_ok=True)

# OUTPUT LABELS
LBL_MB = 1
LBL_AC = 2
LBL_MT = 3
LBL_CP = 4
LBL_MP = 5
# LBL_BR = 6

##### Main procedure

# set_stomos = SetTomos()
vx_um3 = (VOI_VSIZE * 1e-4) ** 3

# Preparing intermediate directories
lio.clean_dir(TEM_DIR)
lio.clean_dir(TOMOS_DIR)

# Start timing total generation
total_start_time = time.time()

for tomo_id in range(NTOMOS):
    print("GENERATING TOMOGRAM NUMBER:", tomo_id)
    hold_time = time.time()

    # tomo = Tomogram(
    #     id=tomo_id,
    #     shape=VOI_SHAPE,
    #     v_size=VOI_VSIZE,
    #     offset=VOI_OFFS
    # )
    #
    # # Generating membranes and adding them to the tomogram
    # for memb_file in MEMBRANES_LIST:
    #     print(f"Generating membranes from file: {memb_file}")
    #     memb_f = MbFile()
    #     file_path = os.path.join(ROOT_PATH, memb_file)
    #     params = memb_f.load_mb_file(file_path)
    #     mem_cf = memb_f.den_cf
    #
    #     # Adding a set of membranes to the tomogram
    #     tomo.gen_set_mbs(
    #         params=params,
    #         cf=mem_cf,
    #         verbosity=True
    #     )
    # 
    # print("Tomogram generation time (s):", time.time() - hold_time, "\n")
    #
    # # Saving tomogram density and labels
    # write_mrc_path = TOMOS_DIR / f"tomo_{tomo_id}_den.mrc"
    # lio.write_mrc(
    #     tomo.density,
    #     write_mrc_path,
    #     v_size=VOI_VSIZE,
    #     dtype=np.float32
    # )

    tomo = SynthTomo(
        id=tomo_id,
        mbs_file_list=MEMBRANES_LIST,
        hns_file_list=[],
        pns_file_list=PROTEINS_LIST,
        pms_file_list=[],
    )

    tomo.gen_sample(
        data_path=ROOT_PATH,
        shape=VOI_SHAPE,
        v_size=VOI_VSIZE,
        offset=VOI_OFFS,
        verbosity=True
    )
    
    print("Tomogram generation time (s):", time.time() - hold_time, "\n")

    # write_mrc_path = TOMOS_DIR / f"tomo_{tomo_id}_den.mrc"
    # lio.write_mrc(
    #     tomo.sample.density,
    #     write_mrc_path,
    #     v_size=VOI_VSIZE,
    #     dtype=np.float32
    # )

    # Saving tomogram density and labels
    tomo.save_tomo(output_folder=TOMOS_DIR)


# Print total generation time
total_time = time.time() - total_start_time
total_minutes = int(total_time // 60)
total_seconds = total_time % 60
avg_time = total_time / NTOMOS
avg_minutes = int(avg_time // 60)
avg_seconds = avg_time % 60

print("\n" + "="*60)
print(f"TOTAL GENERATION TIME: {total_minutes}min {total_seconds:.2f}s ({total_time:.2f}s)")
print(f"Average time per tomogram: {avg_minutes}min {avg_seconds:.2f}s ({avg_time:.2f}s)")
print(f"Total tomograms generated: {NTOMOS}")
print("="*60 + "\n")


# # Loop for tomograms
# for tomod_id in range(NTOMOS):

#     print("GENERATING TOMOGRAM NUMBER:", tomod_id)
#     hold_time = time.time()

#     # Generate the VOI and tomogram density
#     if isinstance(VOI_SHAPE, str):
#         voi = lio.load_mrc(VOI_SHAPE) > 0
#         voi_off = np.zeros(shape=voi.shape, dtype=bool)
#         voi_off[
#             VOI_OFFS[0][0] : VOI_OFFS[0][1],
#             VOI_OFFS[1][0] : VOI_OFFS[1][1],
#             VOI_OFFS[2][0] : VOI_OFFS[2][1],
#         ] = True
#         voi = np.logical_and(voi, voi_off)
#         del voi_off
#     else:
#         voi = np.zeros(shape=VOI_SHAPE, dtype=bool)
#         voi[
#             VOI_OFFS[0][0] : VOI_OFFS[0][1],
#             VOI_OFFS[1][0] : VOI_OFFS[1][1],
#             VOI_OFFS[2][0] : VOI_OFFS[2][1],
#         ] = True
#         voi_inital_invert = np.invert(voi)
#     voi_voxels = voi.sum()
#     tomo_lbls = np.zeros(shape=VOI_SHAPE, dtype=np.float32)
#     tomo_den = np.zeros(shape=voi.shape, dtype=np.float32)
#     synth_tomo = SynthTomo()
#     poly_vtp, mbs_vtp, skel_vtp = None, None, None
#     entity_id = 1
#     mb_voxels, ac_voxels, mt_voxels, cp_voxels, mp_voxels = 0, 0, 0, 0, 0
#     set_mbs = None

#     # Membranes loop
#     count_mbs, hold_den = 0, None
#     for p_id, p_file in enumerate(MEMBRANES_LIST):

#         print("\tPROCESSING FILE:", p_file)

#         # Loading the membrane file
#         memb_f = MbFile()
#         file_path = os.path.join(ROOT_PATH, p_file)
#         params = memb_f.load_mb_file(file_path)

#         # Obtaining membrane generator from factory
#         mb_type = params["MB_TYPE"]
#         memb_g = MbFactory.create(mb_type, params)

#         # Creating the set of membranes
#         set_mbs = SetMembranes(
#             voi,
#             VOI_VSIZE,
#             memb_g
#         )

#         # Building the set of membranes
#         set_mbs.build_set(verbosity=True)
#         hold_den = set_mbs.tomo

#         # Applying density contrast factor
#         den_cf = memb_f.den_cf
#         if den_cf is not None:
#             hold_den *= den_cf

#         # Density tomogram updating
#         voi = set_mbs.voi
#         mb_mask = set_mbs.tomo > 0
#         mb_mask[voi_inital_invert] = False
#         tomo_lbls[mb_mask] = entity_id
#         count_mbs += set_mbs.num_mbs
#         mb_voxels += (tomo_lbls == entity_id).sum()
#         tomo_den = np.maximum(tomo_den, hold_den)
#         hold_vtp = set_mbs.vtp
#         pp.add_label_to_poly(hold_vtp, entity_id, "Entity", mode="both")
#         pp.add_label_to_poly(hold_vtp, LBL_MB, "Type", mode="both")
#         if poly_vtp is None:
#             poly_vtp = hold_vtp
#             skel_vtp = hold_vtp
#         else:
#             poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
#             skel_vtp = pp.merge_polys(skel_vtp, hold_vtp)
#         synth_tomo.add_set_mbs(set_mbs=set_mbs, m_type="Membrane", lbl=entity_id, code=memb_f.type)
#     entity_id += 1

#     write_mrc_path = TOMOS_DIR / ("tomo_" + str(tomod_id) + "_den.mrc")
#     lio.write_mrc(tomo_den, write_mrc_path, v_size=VOI_VSIZE, dtype=np.float32)
