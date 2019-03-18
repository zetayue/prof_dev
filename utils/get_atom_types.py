import os


def get_atom_types():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data")
    train_path = os.path.join(data_path, "CASP_SCWRL")
    dir_paths = os.scandir(train_path)
    atoms = {}
    resi = {}
    atom_label = 0.0
    resi_label = 0.0
    for d in dir_paths:
        if not d.name.startswith("T"):
            continue
        files = os.scandir(d.path)
        f = next(files)
        while f.name == "list.dat" or f.name.endswith("pdb"):
            f = next(files)
        with open(f, "r") as df:
            line = df.readline().rstrip()
            while line and line != "TER":
                if line.split()[2] not in atoms.keys():
                    atoms[line.split()[2]] = atom_label / 61.0
                    atom_label += 1
                if line.split()[3] not in resi.keys():
                    resi[line.split()[3]] = resi_label / 19.0
                    resi_label += 1
                line = df.readline().rstrip()
    print(atoms)
    print(resi)


if __name__ == "__main__":
    get_atom_types()
