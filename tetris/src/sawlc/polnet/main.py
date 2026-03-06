import random

from pathlib import Path

from .tomogram import SynthTomo

def gen_tomos(config: dict) -> None:
    root_dir = config["folders"]["root"]
    if root_dir is None:
        root_dir = Path(__file__).parents[2]
    data_rpath = config["folders"]["input"]
    data_apath = root_dir / data_rpath
    if not data_apath.exists():
        raise FileNotFoundError(f"Data path {data_apath} does not exist.")
    out_rpath = config["folders"]["output"]
    out_apath = root_dir / out_rpath
    if not out_apath.exists():
        out_apath.mkdir(parents=True, exist_ok=True)

    seed = config["global"].get("seed", None)
    n_tomos = config["global"]["ntomos"]

    voi_shape = config["sample"]["voi_shape"]
    voi_offs = config["sample"]["voi_offset"]
    vx_size = config["sample"]["vx_size"]
    membranes = config["sample"].get("membranes", [])
    helices = config["sample"].get("helices", [])
    proteins = config["sample"].get("proteins", [])
    mb_proteins = config["sample"].get("mb_proteins", [])

    random.seed(seed)

    for tomo_id in range(n_tomos):
        synth_tomo = SynthTomo(
            id=tomo_id + 1,
            mbs_file_list=membranes,
            hns_file_list=helices,
            pns_file_list=proteins,
            pms_file_list=mb_proteins
        )

        synth_tomo.gen_sample(
            data_path=data_apath,
            shape=voi_shape,
            v_size=vx_size,
            offset=voi_offs,
            verbosity=True
        )

        synth_tomo.save_tomo(output_folder=out_apath / f"Tomo{tomo_id + 1:03d}")
        synth_tomo.print_summary()




