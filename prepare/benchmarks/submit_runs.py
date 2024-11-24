import argparse
import os
import subprocess
from pathlib import Path

# from helm_experiments.helm_experiments_constants import ScenarioNames, HELM_SCENARIOS
# from helm_experiments.helm_experiments_constants import Constants as HelmConstants

# some parameters related to job submission
MACHINE_TYPE = "x86"
NUM_CORES = 16
NUM_GPUS = 0
MEMORY = "250g"  # amount of memory we will request for the job
# GPU_TYPE = 'v100'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="disable job submission, just print the command to run",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="disable job submission, run the command locally",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="12h",
        choices=["1h", "6h", "12h", "24h"],
        help="the desired CCC job duration",
    )

    parser.add_argument(
        "--models",
        type=str,
    )
    parser.add_argument(
        "--out_path",
        type=str,
    )

    args = parser.parse_args()
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    log_dir = os.path.join(root_dir, "output", "bench", "logs")
    os.makedirs(log_dir, exist_ok=True)

    models = args.models.split(",")
    # for dataset in datasets:
    #     if dataset not in [s.value for s in ScenarioNames]:
    #         raise Exception(f"invalid dataset name '{dataset}', available datasets: {list(ScenarioNames)}")

    experiment_args = {
        k: v
        for k, v in vars(args).items()
        if k not in {"dummy", "local", "duration", "models"}
    }
    fname_to_run = "tables.py"
    full_fname_to_run = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), fname_to_run
    )

    exp_desc = ""
    common_params_str = ""
    for arg_name, arg_value in experiment_args.items():
        if arg_value is False or arg_value is None:
            continue
        elif arg_value is True:
            common_params_str += f"--{arg_name} "
            exp_desc += f"{arg_name}_"
        elif isinstance(arg_value, list):
            common_params_str += f"--{arg_name} " + " ".join(arg_value) + " "
        else:
            common_params_str += f"--{arg_name} {arg_value} "
            exp_desc += f"{arg_name}_{arg_value}_"
    exp_desc = exp_desc[:-1]
    print(exp_desc)

    for model in models:
        run_name = f"{exp_desc}_{model}"
        params_str = common_params_str + f"--models {model} "

        command_to_run = f"PYTHONPATH=. /dccstor/gmc/shir/anaconda3/envs/unitxt-env/bin/python {full_fname_to_run} {params_str}"
        out_fname = os.path.join(log_dir, run_name + "_out.txt")
        err_fname = os.path.join(log_dir, run_name + "_err.txt")

        print(f"duration: {args.duration}")
        print(f"out_fname: {out_fname}")
        print(f"err_fname: {err_fname}")
        print(f"command_to_run: {command_to_run}")

        if args.dummy:
            print("NOTICE: job submission is disabled - job was not submitted")
            continue
        elif args.local:
            subprocess.call(command_to_run.split()[3:])
        else:
            # submit the job
            from cvar_pyutils.ccc import submit_job

            job_id, jbsub_output = submit_job(
                command_to_run=command_to_run,
                machine_type=MACHINE_TYPE,
                duration=args.duration,
                num_cores=NUM_CORES,
                num_gpus=NUM_GPUS,
                name=run_name,
                mem=MEMORY,
                # gpu_type=GPU_TYPE,
                out_file=out_fname,
                err_file=err_fname,
            )

            print(f"job_id: {job_id}")
            print(f"jbsub_output: {jbsub_output}")

            # write to log file
            log_fname = os.path.join(log_dir, run_name + "_log.txt")
            Path(log_fname).parent.mkdir(parents=True, exist_ok=True)
            with open(log_fname, "w") as log_file:
                log_file.write(f"args passed to {__file__}:\n")
                for k, v in args.__dict__.items():
                    log_file.write(f"{k}: {v}\n")

                log_file.write(f"\nsubmitted job id: {job_id}\n\n")
                log_file.write(f"jbsub output: {jbsub_output}\n")
