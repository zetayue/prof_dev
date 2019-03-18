import os
import numpy as np
import sys
import logging
import time
import multiprocessing as mp
import h5py
import freesasa as fs
from utils import provider


def pc_normalize(pc):  # pc = point cloud
    avg = np.mean(pc, axis=0)
    pc -= avg
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc /= m
    return pc


class CASPDataset():

    # list.dat file format in CASP_SCWRL dataset
    list_dat = {
        "decoy": 0,
        "rmsd": 1,
        "tmscore": 2,
        "gdt_ts": 3,
        "gdt_ha": 4
    }

    def __init__(self,
                 batch_size=32,
                 npoints=400,
                 split="train",
                 normalize=True,
                 hdf5=True,
                 normal_channel=False,
                 shuffle=None,
                 cache_size = None):
        self.PLACEHOLDER = float("-inf")
        self.ATOM_TYPES = 28
        self.ATTRIBUTES_EACH_ATOM = 5

        self.root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.root, "data")
        self.training_path = os.path.join(self.data_path, "CASP_SCWRL")
        self.testing_path_stg1 = os.path.join(
            self.data_path, "CASP11Stage1_SCWRL")
        self.testing_path_stg2 = os.path.join(
            self.data_path, "CASP11Stage2_SCWRL")

        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize

        self.hdf5 = hdf5
        assert(split == "train" or split == "test" or split == "validate")
        self.split = split
        if self.split == "train":
            self.hdf5_file = os.path.join(self.data_path, "training_set.hdf5")
        elif self.split == "validate":
            self.hdf5_file = os.path.join(self.data_path, "validation_set.hdf5")
        else:
            self.hdf5_file = os.path.join(self.data_path, "test_set.hdf5")
        if cache_size is None:
            self.cache_size = batch_size * 3
        else:
            self.cache_size = cache_size

        self.train_sets = os.path.join(
            self.training_path, "DescriptionClean", "training_set.dat")
        self.validate_sets = os.path.join(
            self.training_path, "DescriptionClean", "validation_set.dat")

        self.normal_channel = normal_channel

        if shuffle is None:
            if split == "train":
                self.shuffle = True
            else:
                self.shuffle = False

        logging.info("Creating data set inventory...")
        self.reset(hdf5=self.hdf5)
        logging.info("Loading data into cache...")
        self.cache_data()

    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(
                rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(
                rotated_data)

        jittered_data = provider.random_scale_point_cloud(
            rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    def create_atom_positions(self):
        # Example:
        # atom_positions = {
        #    "ALA": {
        #       N: [occupancy, x, y, z, sol_area],
        #       CA: [occupancy, x, y, z, sol_area],
        #       C: [occupancy, x, y, z, sol_area],
        #       O: [occupancy, x, y, z, sol_area],
        #       CB: [occupancy, x, y, z, sol_area]
        #    }
        # }
        order = iter(
            range(self.ATOM_TYPES * self.ATTRIBUTES_EACH_ATOM))  # 28 * 5
        residues = ['MET', 'LEU', 'ILE', 'SER', 'HIS', 'ASP', 'ASN',
                    'GLN', 'LYS', 'ALA', 'GLY', 'PHE', 'THR', 'GLU', 'CYS',
                    'TRP', 'PRO', 'TYR', 'VAL', 'ARG']
        atom_positions = {}
        # add N, CA, C, O
        n_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        ca_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        c_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        o_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        for resi in residues:
            atom_positions[resi] = dict(
                N=n_pos,
                CA=ca_pos,
                C=c_pos,
                O=o_pos
            )
        # add CB
        cb_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        for key in atom_positions.keys():
            if key != "GLY":
                atom_positions[key]["CB"] = cb_pos
        # add position G
        cg_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        cg2_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        for key in atom_positions.keys():
            if key == "CYS":
                atom_positions[key]["SG"] = cg_pos
            elif key == "SER":
                atom_positions[key]["OG"] = cg2_pos
            elif key == "ILE" or key == "VAL":
                atom_positions[key]["CG1"] = cg_pos
                atom_positions[key]["CG2"] = cg2_pos
            elif key == "THR":
                atom_positions[key]["OG1"] = cg2_pos
                atom_positions[key]["CG2"] = cg_pos
            elif key != "ALA" or key != "GLY":
                atom_positions[key]["CG"] = cg_pos
        # add position D
        d1_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        d2_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        d3_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        d4_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        for key in atom_positions.keys():
            if key == "LEU":
                atom_positions[key]["CD1"] = d1_pos
                atom_positions[key]["CD2"] = d2_pos
            elif key in ["GLU", "ILE", "LYS", "PRO", "GLN", "ARG"]:
                atom_positions[key]["CD"] = d1_pos
                atom_positions[key]["CD1"] = d2_pos
            elif key == "MET":
                atom_positions[key]["SD"] = d1_pos
            elif key in ["PHE", "TRP", "TYR", "HIS"]:
                atom_positions[key]["CD1"] = d3_pos
                atom_positions[key]["CD2"] = d4_pos
        atom_positions["HIS"]["ND1"] = d3_pos
        atom_positions["ASP"]["OD1"] = d3_pos
        atom_positions["ASP"]["OD2"] = d4_pos
        atom_positions["ASN"]["OD1"] = d3_pos
        atom_positions["ASN"]["ND2"] = d4_pos
        # add positon E
        e1_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e2_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e3_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e4_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e5_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e6_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e7_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e8_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        e9_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        atom_positions["LYS"]["CE"] = e1_pos
        atom_positions["ARG"]["NE"] = e1_pos
        atom_positions["HIS"]["CE1"] = e2_pos
        atom_positions["HIS"]["NE2"] = e3_pos
        atom_positions["TRP"]["NE1"] = e3_pos
        atom_positions["TRP"]["CE2"] = e2_pos
        atom_positions["GLU"]["OE1"] = e4_pos
        atom_positions["GLU"]["OE2"] = e5_pos
        atom_positions["GLN"]["OE1"] = e4_pos
        atom_positions["GLN"]["NE2"] = e5_pos
        for key in ["PHE", "TYR"]:
            atom_positions[key]["CE1"] = e6_pos
            atom_positions[key]["CE2"] = e7_pos
        atom_positions["MET"]["CE"] = e8_pos
        atom_positions["TRP"]["CE3"] = e9_pos
        # add position Z
        z1_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        z2_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        z3_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        z4_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        atom_positions["LYS"]["NZ"] = z1_pos
        atom_positions["ARG"]["CZ"] = z1_pos
        atom_positions["PHE"]["CZ"] = z2_pos
        atom_positions["TYR"]["CZ"] = z2_pos
        atom_positions["TRP"]["CZ2"] = z3_pos
        atom_positions["TRP"]["CZ3"] = z4_pos
        # add position H
        h1_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        h2_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        h3_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        h4_pos = [next(order) for _ in range(self.ATTRIBUTES_EACH_ATOM)]
        atom_positions["ARG"]["NH1"] = h1_pos
        atom_positions["ARG"]["NH2"] = h2_pos
        atom_positions["TRP"]["CH2"] = h3_pos
        atom_positions["TYR"]["OH"] = h4_pos
        return atom_positions

    def _put_atom_src(self, atom, residue, solvent_access, atom_positions, atom_to_num):
        atom_name = atom[12:16].strip().upper()
        resi_name = atom[17:20].strip().upper()
        atom_num = int(atom[6:11])
        atom_type = atom.strip().split()[-1]
        x = float(atom[30:38])
        y = float(atom[38:46])
        z = float(atom[46:54])
        try:
            sol_area = solvent_access.atomArea(atom_num)
        except AssertionError:
            sol_area = 0.0

        try:
            coor_pos = atom_positions[resi_name][atom_name]
            residue[coor_pos[0]] = atom_to_num[atom_type]
            residue[coor_pos[1]:coor_pos[3]+1] = x, y, z
            residue[coor_pos[4]] = sol_area

        except:
            # logging.debug("Error occur.")
            # logging.debug("resi name: {}, type: {}".format(resi_name, type(resi_name)))
            # logging.debug("atom name: {}".format(atom_name))
            pass

        finally:
            return residue

    def _put_atom_hdf5(self, atom, residue, atom_positions, atom_to_num):
        atom_name = atom["atom_name"].upper()
        resi_name = atom["resi_name"].upper()
        atom_type = atom["atom_type"]
        x = atom["x"]
        y = atom["y"]
        z = atom["z"]
        sol_area = atom["sol_area"]

        try:
            coor_pos = atom_positions[resi_name][atom_name]
            residue[coor_pos[0]] = atom_to_num[atom_type]
            residue[coor_pos[1]:coor_pos[3]+1] = x, y, z
            residue[coor_pos[4]] = sol_area

        except:
            # logging.debug("Error occur.")
            # logging.debug("resi name: {}, type: {}".format(resi_name, type(resi_name)))
            # logging.debug("atom name: {}".format(atom_name))
            pass

    def build_residue(self):
        return [0, 0, 0, 0] * 5 + ([0] + [self.PLACEHOLDER] * 3 + [0]) * 24

    def _get_item_src(self, decoy):
        """
        decoy: str, path to the decoy
        """
        atom_to_num = {
            "C": 1,
            "N": 2,
            "O": 3,
            "S": 4
        }
        residues = []
        atom_positions = self.create_atom_positions()
        residue = self.build_residue()
        structure = fs.Structure(decoy)
        solvent_access = fs.calc(structure)
        with open(decoy, "r") as f:
            line = f.readline().rstrip()
            while not line.startswith("ATOM"):
                line = f.readline().rstrip()
            cur_resi = int(line[22:26])

            # PDB file stardard format
            # COLUMNS   DATA  TYPE    FIELD
            # -------------------------------------------
            #  1 -  6   Record name   "ATOM  "
            #  7 - 11   Integer       Atom serial #
            # 13 - 16   Atom          Atom name
            # 17        Character     Alternate location
            # 18 - 20   Residue name  resName
            # 22        Character     chainID
            # 23 - 26   Integer       resSeq
            # 27        AChar         Code for insertion of residues
            # 31 - 38   Real(8.3)     x
            # 39 - 46   Real(8.3)     y
            # 47 - 54   Real(8.3)     z
            # 55 - 60   Real(6.2)     occupancy
            # 61 - 66   Real(6.2)     tempFactor
            # 77 - 78   LString(2)    element
            # 79 - 80   LString(2)    Charge  on the atom

            while line:
                if line.startswith("TER"):
                    break
                if not line.startswith("ATOM"):
                    line = f.readline().rstrip()
                    continue

                # ignore hydrogens
                atom_type = line[-1]
                if atom_type == "H":
                    line = f.readline().rstrip()
                    continue

                resi_num = int(line[22:26])
                if resi_num > cur_resi:
                    residues.append(residue)
                    if len(residues) == 400:
                        break
                    residue = self.build_residue()
                    cur_resi = resi_num
                residue = self._put_atom_src(
                    line.rstrip(), residue, solvent_access, atom_positions, atom_to_num)
                line = f.readline().rstrip()

        # normalize residues
        pc = np.ones((self.npoints, self.num_channel())) * float("-inf")
        residues = np.array(residues)
        logging.debug("decoy shape: {}".format(residues.shape))
        x_mean = np.mean(residues[:, 1])
        y_mean = np.mean(residues[:, 2])
        z_mean = np.mean(residues[:, 3])
        for i in range(self.num_channel() // self.ATTRIBUTES_EACH_ATOM):
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+1] -= x_mean
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+2] -= y_mean
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+3] -= z_mean
        pc[0:residues.shape[0], :] = residues

        target_path = os.path.dirname(decoy)
        gdt_ts = 0.0
        with open(os.path.join(target_path, "list.dat"), "r") as lst:
            info = lst.readline()
            while info:
                if info.startswith(os.path.basename(decoy)):
                    gdt_ts = float(
                        info.split()[CASPDataset.list_dat["gdt_ts"]])
                    break
                info = lst.readline()

        return pc, gdt_ts

    def write_to_queue(self, decoy):
        with h5py.File(self.hdf5_file, "r") as h5f:
            decoy_datasets = h5f[decoy]
            data = {"gdt_ts": decoy_datasets.attrs["gdt_ts"]}
            for dataset in [
                "x", "y", "z", 
                "atom_name",
                "resi_name", 
                "resi_seq", 
                "atom_type", 
                "sol_area"
            ]:
                data[dataset] = decoy_datasets[dataset][...]
            while self.cache.qsize() >= self.cache_size:
                time.sleep(0.1)
            self.cache.put(data)
    
    def cache_data(self):
        mp_manager = mp.Manager()
        self.cache = mp_manager.Queue()
        p = mp.Pool(2)
        for dataset in self.pc_datasets:
            p.apply_async(self.write_to_queue, (dataset,))
    
    def _get_item_hdf5(self):
        atom_to_num = {
            "C": 1,
            "N": 2,
            "O": 3,
            "S": 4
        }
        residues = []
        atom_positions = self.create_atom_positions()
        residue = self.build_residue()
        data = self.cache.get()
        try:
            len_data = len(data["x"])
        except:
            return None, None
        cur_resi = data["resi_seq"][0]
        for i in range(len_data):
            if data["atom_type"][i] == "H":
                continue
            resi_seq = data["resi_seq"][i]
            if resi_seq > cur_resi:
                residues.append(residue)
                if len(residues) == 400:
                    break
                residue = self.build_residue()
                cur_resi = resi_seq
            atom = {}
            for dataset in data.keys():
                if dataset == "gdt_ts":
                    continue
                atom[dataset] = data[dataset][i]
            self._put_atom_hdf5(atom, residue, atom_positions, atom_to_num)
        
        # normalize residues
        pc = np.zeros((self.npoints, self.num_channel()))
        residues = np.array(residues)
        logging.debug("decoy shape: {}".format(residues.shape))
        x_mean = np.mean(residues[:, 1])
        y_mean = np.mean(residues[:, 2])
        z_mean = np.mean(residues[:, 3])
        for i in range(self.num_channel() // self.ATTRIBUTES_EACH_ATOM):
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+1] -= x_mean
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+2] -= y_mean
            residues[:, self.ATTRIBUTES_EACH_ATOM*i+3] -= z_mean
        pc[0:residues.shape[0], :] = residues

        # get gdt_ts score
        gdt_ts = data["gdt_ts"]
    
        return pc, gdt_ts
    
    def __len__(self):
        return len(self.pc_paths)

    def num_channel(self):
        return self.ATTRIBUTES_EACH_ATOM * self.ATOM_TYPES

    def count_residues(self, filepath):
        with open(filepath, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            line = f.readline()
            offset = 0
            while True:
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                    offset += 1
                line = f.readline()
                if not line.startswith(b"ATOM"):
                    offset += 1
                    f.seek(-offset, os.SEEK_END)
                else:
                    break
            line = line.decode()
            resi_num = int(line[22:26])
            return resi_num

    def reset(self, hdf5=True):
        if hdf5:
            self.pc_datasets = []
            h5f = h5py.File(self.hdf5_file, "r")
            if self.split == "train":
                group = h5f["training"] 
            elif self.split == "validate":
                group = h5f["validation"]
            
            for template in group:
                t_group = group[template]
                for decoy in t_group:
                    self.pc_datasets.append(t_group[decoy].name)
            logging.debug("pc_dataset length: {}".format(len(self.pc_datasets)))
            if self.shuffle:
                np.random.shuffle(self.pc_datasets)
            self.num_batches = (len(self.pc_datasets) +
                                self.batch_size-1) // self.batch_size
            self.batch_idx = 0

            h5f.close()
                
        else:
            self.pc_paths = []
            folders = []
            if self.split == "train":
                with open(self.train_sets, 'r') as train_f:
                    folder = train_f.readline().rstrip()
                    while folder:
                        folders.append(folder)
                        folder = train_f.readline().rstrip()

            elif self.split == "validate":
                with open(self.validate_sets, 'r') as val_f:
                    folder = val_f.readline().rstrip()
                    while folder:
                        folders.append(folder)
                        folder = val_f.readline().rstrip()
            else:
                # placeholder for the test set
                pass

            for folder in folders:
                counted = False
                # exclude description folders
                if not folder.startswith("T"):
                    continue
                items = os.scandir(os.path.join(self.training_path, folder))
                for item in items:
                    if item.name == "list.dat":
                        continue
                    if not counted:
                        resi_num = self.count_residues(item.path)
                        counted = True
                    # discard the protein with length shorter than 50 or longer than 400
                    if resi_num > 400 or resi_num < 51:
                        break
                    self.pc_paths.append(os.path.abspath(item.path))
            logging.debug("total point clouds: {}".format(len(self.pc_paths)))

            self.idxs = np.arange(0, len(self.pc_paths))
            if self.shuffle:
                np.random.shuffle(self.idxs)
            self.num_batches = (len(self.pc_paths) +
                                self.batch_size-1) // self.batch_size
            self.batch_idx = 0
            logging.debug("batch size: {}".format(self.batch_size))
            logging.debug("num of batches: {}".format(self.num_batches))
            logging.debug("index: {}".format(self.idxs))

    def shuffle_dataset(self):
        np.random.shuffle(self.pc_datasets)
    
    def has_next_batch(self):
        # the line below is for code testing
        # return self.batch_idx < 3
        if self.split == "train":
            return self.batch_idx < self.num_batches
            
        elif self.split == "validate":
            return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        batch_data = np.zeros((self.batch_size, self.npoints, self.num_channel()))
        batch_label = np.zeros((self.batch_size), dtype=np.float32)
        if self.hdf5:
            i = 0
            while i < self.batch_size:
                ps, gdt = self._get_item_hdf5()
                if ps is None:
                    continue
                batch_data[i] = ps
                batch_label[i] = gdt
                i += 1
        else:
            start_idx = self.batch_idx * self.batch_size
            end_idx = min((self.batch_idx+1) * self.batch_size, len(self.pc_paths))
            bsize = end_idx - start_idx
            for i in range(bsize):
                decoy_path = self.pc_paths[self.idxs[start_idx+i]]
                ps, gdt = self._get_item_src(decoy_path)
                batch_data[i] = ps
                batch_label[i] = gdt

        self.batch_idx += 1
        batch_label = np.expand_dims(batch_label, 1)
        if augment:
            batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label


class ProcessedDataset():

    def __init__(self, batch_size=32, split="training", shuffle=False):

        self.ATOM_TYPES = 28
        self.ATTRIBUTES_EACH_ATOM = 5
        self.N_POINTS = 400

        self.root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.root, "data")
        self.dataset_path = os.path.join(
            self.data_path, "processed_training.hdf5")

        self.batch_size = batch_size

        assert(split == "training" or split == "validation" or split == "testing")
        self.split = split

        self.shuffle = shuffle

        logging.info("Preparing data set...")
        self.reset()
        logging.info("Done!")

    def _get_item(self, decoy: "int"):
        """
        decoy: str, path to the decoy
        """
        h5f = h5py.File(self.dataset_path, "r")
        group = h5f[self.split]
        pc_ds = group["point_sets"]
        gdt_ds = group["gdt_ts"]
        pc = pc_ds[decoy]
        gdt_ts = gdt_ds[decoy]
        h5f.close()
        return pc, gdt_ts
    
    def __len__(self):
        return len(self.ps_decoys)

    def num_channel(self):
        return self.ATTRIBUTES_EACH_ATOM * self.ATOM_TYPES

    def reset(self):

        h5f = h5py.File(self.dataset_path, "r")
        group = h5f[self.split]
        n_decoys = group["point_sets"].len()
        self.ps_decoys = list(range(n_decoys))
        logging.debug("pc_dataset length: {}".format(n_decoys))
        if self.shuffle:
            np.random.shuffle(self.ps_decoys)
        self.num_batches = (n_decoys + self.batch_size-1) // self.batch_size
        self.batch_idx = 0
        h5f.close()

    def shuffle_dataset(self):
        self.batch_idx = 0
        np.random.shuffle(self.ps_decoys)
    
    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        batch_data = np.zeros((self.batch_size, self.N_POINTS, self.num_channel()))
        batch_label = np.zeros((self.batch_size), dtype=np.float32)
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.ps_decoys))
        bsize = end_idx - start_idx
        for i in range(bsize):
            decoy_id = self.ps_decoys[start_idx+i]
            pc, gdt = self._get_item(decoy_id)
            batch_data[i] = pc
            batch_label[i] = gdt

        self.batch_idx += 1
        batch_label = np.expand_dims(batch_label, 1)
        return batch_data, batch_label


class AtomwiseDataset():

    def __init__(self, batch_size=32, split="training", shuffle=False, label_type="normal"):

        self.PLACEHOLDER = float("-inf")
        self.NUM_CHANNEL = 8
        self.ATOM_TYPES = 5
        self.MAX_ATOMS = 8770

        self.root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.root, "data")
        self.dataset_path = os.path.join(
            self.data_path, "processed_atomwise.hdf5")

        self.batch_size = batch_size

        assert(
            split == "training" or split == "validation" or split == "testing")
        self.split = split

        self.shuffle = shuffle
        self.label_type = label_type

        logging.info("Preparing data set...")
        self.reset()

    def convert_to_onehot(self, value):
        onehot = [0] * 10
        if value < 0:
            onehot[0] = 1
            return onehot
        onehot[min(int(value*10), 9)] = 1
        return onehot

    def _get_item(self, decoy: "int", h5f):
        """
        decoy: str, path to the decoy
        """
        pc_ds = h5f[self.split]
        gdt_ds = h5f[self.split + "_label"]
        pc = pc_ds[decoy]
        if self.label_type == "normal":
            gdt_ts = gdt_ds[decoy]
        elif self.label_type == "onehot":
            gdt_ts = self.convert_to_onehot(gdt_ds[decoy])
        return pc, gdt_ts
    
    def __len__(self):
        return len(self.ps_decoys)

    def reset(self):
        h5f = h5py.File(self.dataset_path, "r")
        self.ps_decoys = list(range(h5f[self.split].len()))
        logging.debug(
            "pc_dataset length: {}".format(h5f[self.split].len()))
        if self.shuffle:
            np.random.shuffle(self.ps_decoys)
        self.num_batches = (h5f[self.split].len() + self.batch_size-1) // self.batch_size
        self.batch_idx = 0
        h5f.close()

    def shuffle_dataset(self):
        self.batch_idx = 0
        np.random.shuffle(self.ps_decoys)
    
    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        h5f = h5py.File(self.dataset_path, "r")
        batch_data = np.ones((self.batch_size, self.MAX_ATOMS, self.NUM_CHANNEL)) * self.PLACEHOLDER
        if self.label_type == "normal":
            batch_label = np.zeros((self.batch_size), dtype=np.float32)
        elif self.label_type == "onehot":
            batch_label = np.zeros((self.batch_size, 10), dtype=np.int8)
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.ps_decoys))
        bsize = end_idx - start_idx
        for i in range(bsize):
            decoy_id = self.ps_decoys[start_idx+i]
            pc, gdt = self._get_item(decoy_id, h5f)
            batch_data[i] = pc
            batch_label[i] = gdt

        self.batch_idx += 1
        if self.label_type == "normal":
            batch_label = np.expand_dims(batch_label, 1)
        h5f.close()
        return batch_data, batch_label


class CoordinateDataset(AtomwiseDataset):
    def __init__(self, batch_size=32, split="training", shuffle=False, label_type="normal"):

        self.PLACEHOLDER = 0
        self.NUM_CHANNEL = 3
        self.MAX_ATOMS = 8770

        self.root = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.root, "data")
        self.dataset_path = os.path.join(
            self.data_path, "processed_coordinates.hdf5")
        self.mask_path = os.path.join(
            self.data_path, "coordinates_mask.hdf5")

        self.batch_size = batch_size

        assert(
            split == "training" or split == "validation" or split == "testing")
        self.split = split

        self.shuffle = shuffle
        self.label_type = label_type

        logging.info("Preparing data set...")
        self.reset()

    def convert_to_onehot(self, value):
        onehot = [0] * 10
        if value < 0:
            onehot[0] = 1
            return onehot
        onehot[min(int(value*10), 9)] = 1
        return onehot

    def _get_item(self, decoy: "int", h5f, mask_h5f):
        """
        decoy: str, path to the decoy
        """
        pc_ds = h5f[self.split]
        gdt_ds = h5f[self.split + "_label"]
        mask_ds = mask_h5f[self.split + "_mask"]
        pc = pc_ds[decoy]
        if self.label_type == "normal":
            gdt_ts = gdt_ds[decoy]
        elif self.label_type == "onehot":
            gdt_ts = self.convert_to_onehot(gdt_ds[decoy])
        mask = mask_ds[decoy]
        return pc, gdt_ts, mask
    
    def __len__(self):
        return len(self.ps_decoys)

    def reset(self):
        h5f = h5py.File(self.dataset_path, "r")
        self.ps_decoys = list(range(h5f[self.split].len()))
        logging.debug(
            "pc_dataset length: {}".format(h5f[self.split].len()))
        if self.shuffle:
            np.random.shuffle(self.ps_decoys)
        self.num_batches = (h5f[self.split].len() + self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def shuffle_dataset(self):
        self.batch_idx = 0
        np.random.shuffle(self.ps_decoys)
    
    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        h5f = h5py.File(self.dataset_path, "r")
        mask_h5f = h5py.File(self.mask_path, "r")
        batch_data = np.zeros(
            (self.batch_size, self.MAX_ATOMS, self.NUM_CHANNEL),
            dtype=np.float32
        )
        batch_mask = np.zeros((self.batch_size, self.MAX_ATOMS), dtype=np.bool)
        if self.label_type == "normal":
            batch_label = np.zeros((self.batch_size), dtype=np.float32)
        elif self.label_type == "onehot":
            batch_label = np.zeros((self.batch_size, 10), dtype=np.int8)
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.ps_decoys))
        bsize = end_idx - start_idx
        for i in range(bsize):
            decoy_id = self.ps_decoys[start_idx+i]
            pc, gdt, mask = self._get_item(decoy_id, h5f, mask_h5f)
            batch_data[i] = pc
            batch_label[i] = gdt
            batch_mask[i] = mask

        self.batch_idx += 1
        if self.label_type == "normal":
            batch_label = np.expand_dims(batch_label, 1)
        h5f.close()
        mask_h5f.close()
        return batch_data, batch_label, batch_mask


if __name__ == "__main__":
    # counter = 0
    # logging.basicConfig(level=logging.INFO)
    dataset = CoordinateDataset(split="training", shuffle=True, label_type="normal")
    h5f = h5py.File(dataset.dataset_path, "r")
    mask_h5f = h5py.File(dataset.mask_path, "r")
    
    pc, gdt, mask = dataset.next_batch()

    print("pc: {}".format(pc))
    print("gdt_ts: {}".format(gdt))
    print(f"mask: {mask}")
    h5f.close()
    mask_h5f.close()
    # tests = [0.01, 0.12, 0.22, 0.35, 0.77, 0.99, 1, -1, -2, 2]
    # results = []
    # for t in tests:
    #     results.append(dataset.convert_to_onehot(t))
    
    # print("test: {}".format(tests))
    # print("results: {}".format(results))

    # batch, gdt = dataset.next_batch()
    # print(f"batch shape: {batch.shape}")
    # print(f"gdt: {gdt}")
    # print(f"gdt shape: {gdt.shape}")
