# %%
import json
import numpy as np
from scipy.spatial.distance import cdist
from ase import Atoms
from ase.io.vasp import read_vasp, write_vasp
from ase.io.espresso import read_espresso_in, read_espresso_out
import datetime
import pandas as pd
from tabulate import tabulate
import pprint
from ase.units import create_units

units = create_units("2006")

import os

# %%
config = {"type": "vasp", "name": "structure", "path": "structure.vasp", "save_pbc_vasp": False, "read_all_steps": False}

with open("./config.json", "r") as file:
    config_load = json.load(file)

config = {key: config_load.get(key, value) for key, value in config.items()}

# %%
with open("./born_effective_charges.json", "r") as file:
    zstars = json.load(file)

b_sites = list(zstars["B-site"].keys())
a_sites = list(zstars["A-site"].keys())


# %%
def pbc(positions, cell, nat):
    n = 3 * 3 * 3

    positions_pbc = np.empty(shape=(nat * n, 7))

    k = 0
    for i_x in range(-1, 2):
        for i_y in range(-1, 2):
            for i_z in range(-1, 2):
                lp = cell.copy()
                lp *= np.array([i_x, i_y, i_z])[:, np.newaxis]
                disp_vec = lp.sum(axis=0)
                positions_pbc[nat * k : nat * (k + 1), 1:4] = i_x, i_y, i_z
                positions_pbc[nat * k : nat * (k + 1), 4:] = positions + disp_vec
                k += 1

    positions_pbc[:, 0] = np.tile(np.arange(nat), 27)

    cell_pbc = cell * [3, 3, 3]
    return (positions_pbc, cell_pbc)


# %%
def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2

    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 180:
        angle_deg = 360 - angle_deg

    return angle_deg


# %%
def calculate_polarization(Z_star, displacements, volume):
    e = 1.602176634e-19  # elementary charge in C

    # Convert Ångström to meters
    displacements_meters = displacements * 1.0e-10
    # Convert Ångström³ to meters³
    volume_meters3 = volume * 1.0e-30

    polarization_x = e / volume_meters3 * Z_star * displacements_meters[0]
    polarization_y = e / volume_meters3 * Z_star * displacements_meters[1]
    polarization_z = e / volume_meters3 * Z_star * displacements_meters[2]

    return np.array([polarization_x, polarization_y, polarization_z])


# %%
class Structure:
    def __init__(self, obj, config):
        self.obj = obj
        self.coords_cartes = obj.get_positions()
        self.coords_direct = obj.get_scaled_positions()
        self.nat = obj.get_global_number_of_atoms()
        self.symbols = obj.get_chemical_symbols()
        self.cell = np.array(obj.get_cell())
        self.V = obj.get_volume()
        self.config = config
        self.config["outname"] = config["name"] + ".out"

    def get_pbc(self):
        self.pbc_positions, self.cell_pbc = pbc(self.coords_cartes, self.cell, self.nat)
        self.coords_cartes_pbc = self.pbc_positions[:, 4:]
        self.symbols_pbc = np.tile(np.array(self.obj.symbols), 27)
        self.obj_pbc = Atoms(positions=self.coords_cartes_pbc, cell=self.cell_pbc, symbols=self.symbols_pbc)

    def get_disp_pol_alpha(self):
        bool_o = self.symbols_pbc == "O"
        bool_cat = self.symbols_pbc != "O"
        bool_origin = (self.pbc_positions[:, 1] == 0) & (self.pbc_positions[:, 2] == 0) & (self.pbc_positions[:, 3] == 0)

        oxygens = self.coords_cartes_pbc[bool_o]
        cations = self.coords_cartes_pbc[bool_origin & bool_cat]
        cations_symbols = self.symbols_pbc[bool_origin & bool_cat]

        dists = cdist(cations, oxygens)
        V = self.V

        disp_pol_alpha = {}
        P = np.empty(shape=(len(cations), 3))
        D = np.empty(shape=(len(cations), 3))
        alphas = []

        for i in range(len(cations)):
            symbol = cations_symbols[i]
            cation_coords = cations[i]
            if symbol in a_sites:
                # A = True
                zstar = zstars["A-site"][symbol]
                hedra_idx = np.argpartition(dists[i], 12)[:12]
                headra_coords = oxygens[hedra_idx]
                center = headra_coords.mean(axis=0)
                d = cation_coords - center
                p = calculate_polarization(zstar, d, V)
                disp_pol_alpha[("A", symbol, i)] = {"p": p, "d": d, "hedra_idx": hedra_idx, "hedra": headra_coords, "alpha": np.array([0, 0, 0])}
                P[i, :] = p
                D[i, :] = d

            elif symbol in b_sites:
                zstar = zstars["B-site"][symbol]
                hedra_idx = np.argpartition(dists[i], 6)[:6]
                headra_coords = oxygens[hedra_idx]
                neis_x = headra_coords[headra_coords[:, 0].argsort()[[0, -1]]]
                neis_y = headra_coords[headra_coords[:, 1].argsort()[[0, -1]]]
                neis_z = headra_coords[headra_coords[:, 2].argsort()[[0, -1]]]
                alpha_x = calculate_angle(neis_x[0], cation_coords, neis_x[1])
                alpha_y = calculate_angle(neis_y[0], cation_coords, neis_y[1])
                alpha_z = calculate_angle(neis_z[0], cation_coords, neis_z[1])
                alpha = np.array([alpha_x, alpha_y, alpha_z])
                center_x = np.mean(neis_x, axis=0)[0]
                center_y = np.mean(neis_y, axis=0)[1]
                center_z = np.mean(neis_z, axis=0)[2]
                center = np.array([center_x, center_y, center_z])
                d = cation_coords - center
                p = calculate_polarization(zstar, d, V)
                disp_pol_alpha[("B", symbol, i)] = {"p": p, "d": d, "hedra_idx": hedra_idx, "hedra": headra_coords, "alpha": alpha}
                P[i, :] = p
                D[i, :] = d
                alphas.append(alpha)
            else:
                print(f"{cations_symbols[i]} not in A/B-sites")

        self.disp_pol_alpha = disp_pol_alpha
        alphas = np.stack(alphas)
        alphas = np.append(alphas, alphas.min(axis=1).reshape(-1, 1), axis=1)
        self.alphas = alphas
        self.alphas_stat = np.array([np.min(alphas[:, -1]), np.mean(alphas[:, -1]), np.max(alphas[:, -1])])
        P_tot = P.sum(axis=0)
        D_tot = D.sum(axis=0)
        self.P_tot = np.append(P_tot, np.sqrt(np.sum(P_tot**2)))
        self.D_tot = np.append(D_tot, np.sqrt(np.sum(D_tot**2)))

        self.P = np.concatenate([P, np.sum(P**2, axis=1).reshape(-1, 1)], axis=1)
        self.D = np.concatenate([D, np.sum(D**2, axis=1).reshape(-1, 1)], axis=1)

    def get_df(self):
        df = pd.DataFrame([[*key, *value["p"], *value["d"], *value["alpha"]] for key, value in self.disp_pol_alpha.items()])
        df.columns = ["site", "cation", "cation_idx", "p_x", "p_y", "p_z", "d_x", "d_y", "d_z", "alpha_x", "alpha_y", "alpha_z"]
        df_grp = df.groupby(["site", "cation"]).describe()
        df_p = df_grp.loc[:, (["p_x", "p_y", "p_z"], ["mean", "std", "min", "max"])]
        df_d = df_grp.loc[:, (["d_x", "d_y", "d_z"], ["mean", "std", "min", "max"])]
        df_alpha = df_grp.loc[:, (["alpha_x", "alpha_y", "alpha_z"], ["mean", "std", "min", "max"])]

        self.df = df
        self.df_p = df_p
        self.df_d = df_d
        self.df_alpha = df_alpha

    def print_log(self):
        now = datetime.datetime.now()

        table = [["", "p_x", "p_y", "p_z", "p_tot", "d_x", "d_y", "d_z", "d_tot", "alpha_max", "alpha_avg", "alpha_min"], ["!", *self.P_tot.round(7), *self.D_tot.round(7), *(180 - self.alphas_stat).round(7)]]
        with open(self.config["outname"], "w") as log_file:
            print("Output Data:", self.config["name"], now.strftime("%Y-%m-%d %H:%M:%S"), file=log_file)
            print("", file=log_file)
            print("", file=log_file)
            print("Summary Table", file=log_file)
            print(tabulate(table, tablefmt="psql"), file=log_file)
            print("", file=log_file)
            print("", file=log_file)
            print("Polarization Statistics Table", file=log_file)
            print(tabulate(self.df_p.T, headers="keys", tablefmt="psql"), file=log_file)
            print("", file=log_file)
            print("Displacement Statistics Table", file=log_file)
            print(tabulate(self.df_d.T, headers="keys", tablefmt="psql"), file=log_file)
            print("", file=log_file)
            print("Alpha Statistics Table", file=log_file)
            print(tabulate(self.df_alpha.xs("B", level=0).T, headers="keys", tablefmt="psql"), file=log_file)

            print("", file=log_file)
            print("", file=log_file)
            print("Verbose Summary", file=log_file)

            for x in self.disp_pol_alpha:
                print(x, file=log_file)
                pprint.pprint(self.disp_pol_alpha[x], log_file, sort_dicts=False, indent=5)
                print("", file=log_file)


# %%
def parse_espresso_out(path, read_all):
    with open(path, "r") as f:
        out = read_espresso_out(f, index=slice(None))
        objects = []
        for item in out:
            objects.append(item)
        energies = [x.get_total_energy() for x in objects]
        idx_min = np.argmin(energies)
    if read_all:
        return objects, objects[idx_min]
    else:
        return objects[idx_min], None


# %%
if config["type"] == "vasp":
    obj = read_vasp(config["path"])
elif config["type"] == "qe_in":
    obj = read_espresso_in(config["path"])
elif config["type"] == "qe_out":
    obj = parse_espresso_out(config["path"], config["read_all_steps"])

# %%
if obj[1] == None:
    struct = Structure(obj[0], config)
    struct.get_pbc()
    if config["save_pbc_vasp"]:
        write_vasp(config["name"] + "_pbc.vasp", struct.obj_pbc, sort=True)
    if config["type"] != "vasp":
        write_vasp(config["name"] + ".vasp", struct.obj, sort=True)
    struct.get_disp_pol_alpha()
    struct.get_df()
    struct.print_log()
else:
    folder_name = config["name"]
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist
    for i, item in enumerate(obj[0]):
        struct = Structure(item, config)
        config["outname"] = f"{folder_name}/{config['name']}_{i+1}.out"
        struct.get_pbc()
        if config["save_pbc_vasp"]:
            write_vasp(f"{folder_name}/{config['name']}_pbc_{i+1}.vasp", struct.obj_pbc, sort=True)
        if config["type"] != "vasp":
            write_vasp(f"{folder_name}/{config['name']}_{i+1}.vasp", struct.obj, sort=True)
        struct.get_disp_pol_alpha()
        struct.get_df()
        struct.print_log()

    struct = Structure(obj[1], config)
    config["outname"] = f"{folder_name}/{config['name']}_min.out"
    struct.get_pbc()
    if config["save_pbc_vasp"]:
        write_vasp(f"{folder_name}/{config['name']}_pbc_min.vasp", struct.obj_pbc, sort=True)
    if config["type"] != "vasp":
        write_vasp(f"{folder_name}/{config['name']}_min.vasp", struct.obj, sort=True)
    struct.get_disp_pol_alpha()
    struct.get_df()
    struct.print_log()
