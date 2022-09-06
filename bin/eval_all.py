"""Can be used to evaluate multiple models that are already pretrained.

Use as `python bin/eval_all.py "in100_dissl_e100*_dir" -s _z8192_mask -d data`
"""


import time
from pathlib import Path
import argparse
import subprocess

MAIN_DIR = Path(__file__).absolute().parents[1]

parser = argparse.ArgumentParser(description='Evaluates all desired models.')
parser.add_argument('pattern', type=str, help='pattern of directories to evaluate')
parser.add_argument("-s", '--sffx', type=str, default="",
                    help='sffx to give to eval_nlprun')
parser.add_argument("-d", '--data', type=str, default="imagenet256",
                    help='data on which to evaluate')

args = parser.parse_args()

dir_to_eval = []

cmd = "squeue --format='%.100j' -u yanndubs".split()
proc1 = subprocess.run(cmd, stdout=subprocess.PIPE)
out = proc1.stdout.decode("utf-8") .replace('\n', ',').replace(' ', '').replace("'", '')
out = out.split(',')[1:-1]  # remove name and last empty string

for p in MAIN_DIR.glob(args.pattern):
    if p.is_dir():
        is_evaluated = len(list(p.glob("eval_w*"))) > 0
        is_pretrained = len(list(p.glob("checkpoints/model_final_checkpoint_phase*.torch"))) > 0
        if is_pretrained and not is_evaluated:
            running_name = f"eval_{p.name}{args.sffx}_{args.data}"
            if running_name in out:
                print(f"{p.name} not evaluated but already running.")
            else:
                dir_to_eval.append(p.name)

for d in dir_to_eval:
    cmd = f"bin/eval_nlprun.sh {d} {args.sffx} {args.data}"
    subprocess.run(cmd.split())
    print()
    time.sleep(3)