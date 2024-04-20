import os
import pymeshlab


ms = pymeshlab.MeshSet()
nphm_folder = '/home/rachmadio/dev/data/NPHM/scan'

folders = sorted(os.listdir(nphm_folder))
for folder in folders:
    scan_folder = os.path.join(nphm_folder, folder, '000')
    load_file = 'scan.ply'
    save_file = 'scan.obj'
    ms.load_new_mesh(os.path.join(scan_folder, load_file))
    ms.save_current_mesh(os.path.join(scan_folder, save_file))