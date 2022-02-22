import logging
import os
import platform
from pathlib import Path
import shutil


def remove_rf(path, not_exist_ok = False):
    """Remove a file or a folder"""
    path = Path(path)

    if not path.exists() and not_exist_ok:
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir:
        shutil.rmtree(path)



def get_nlp_path(user="yanndubs") :
    """Return (create if needed) path on current machine for NLP stanford."""
    machine_name = platform.node().split(".")[0]
    machine_path = Path(f"/{machine_name}/")

    user_paths = list(machine_path.glob(f"*/{user}"))
    if len(user_paths) == 0:
        possible_paths = [p for p in machine_path.iterdir() if "scr" in str(p)]
        user_path = possible_paths[-1] / user
        user_path.mkdir()
    else:
        user_path = user_paths[-1]

    return user_path


def set_nlp_cluster(project_name="vissl"):
    """Set the cluster for NLP stanford."""

    user_path = get_nlp_path()

    # project path on current machine
    proj_path = user_path / project_name
    proj_path.mkdir(exist_ok=True)
    new_data_path = proj_path / "data"

    if not new_data_path.is_symlink():
        # make sure it's a symlink
        remove_rf(new_data_path, not_exist_ok=True)

        if (proj_path.parents[2] / "scr0").exists():
            # if possible symlink to scr0 as that's where most data is saved
            new_data_path.symlink_to(proj_path.parents[2] / "scr0")
        else:
            # if not just use current scr
            new_data_path.symlink_to(proj_path.parents[1])

    prev_work_dir = os.getcwd()
    os.chdir(proj_path)  # change directory on current machine

    return prev_work_dir


