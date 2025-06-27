## Simulation studies
This folder contains all the scripts that trained models. These scripts were written at different times during the development of the hmpinn library. For this reason the configuration files have different formats. Nevertheless, the scripts have been updated to run on the most recent version of the library and all the data produced is compatible. Here are some notes about running these scripts.

### Slurm cluster configurations
Some of the configuration files were written with the purpose of running on slurm clusters. To run the scripts locally the following lines of the configuration files have to be commented out.

```yaml
override hydra/launcher: submitit_slurm
```

```yaml
  launcher:
    # submitit_folder: ${hydra.sweep.dir}/.submitit/%j
    timeout_min: 2880
    cpus_per_task: null
    gpus_per_node: null
    tasks_per_node: 1
    mem_gb: null
    nodes: 1
    name: ${hydra.job.name}
    _target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
    partition: null
    qos: "normal"
    comment: null
    constraint: null
    exclude: null
    gres: gpu:1
    cpus_per_gpu: 1
    gpus_per_task: 1
    mem_per_gpu: null
    mem_per_cpu: null
    account: null
    signal_delay_s: 120
    max_num_timeout: 0
    additional_parameters: {}
    array_parallelism: 256
    setup: null
  ```

The scripts that have this configuration are
- [benchmark](./benchmark/)
- [final_training](./final_training/)
- [harmonic_maps](./harmonic_maps/)
- [weakly_enforcing_BC](./weakly_enforcing_BC/)


### Hydra multirun
Other configurations are intended to be used with [hydra multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) using the [Joblib Launcher](https://hydra.cc/docs/plugins/joblib_launcher/). To run these scripts, the `-m` flag has to be added (e.g. `>>>python myfile.py -m`).
The scripts that have this configuration are
- [best_model_architecture](./best_model_architecture/)
- [composite_loss_simulation](./composite_loss_simulation/)
- [model_v0_simulation](./model_v0_simulation/)
- [model_v1_simulation](./model_v1_simulation/)
