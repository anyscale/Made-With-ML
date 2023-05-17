## Create project

```bash
> anyscale project create -n mlops-course
Authenticating

Output
(anyscale +2.5s) Created project with name mlops-course at https://console.anyscale.com/projects/prj_7b48zz24ac275yc4j3emjmz55p.
(anyscale +2.5s) Please specify the project id as prj_7b48zz24ac275yc4j3emjmz55p when calling Anyscale CLI or SDK commands or Ray Client.
```

Project id is now `prj_7b48zz24ac275yc4j3emjmz55p`. Put into `job_train.yaml`.

## Build cluster env


```bash
> anyscale cluster-env build cluster_env.yaml
Authenticating

Output
(anyscale +2.1s) Creating new cluster environment cli_cluster_env_17_05_23_10_00_38_415984
(anyscale +2.9s) Waiting for cluster environment to build. View progress at https://console.anyscale.com/configurations/app-config-details/bld_zexskbdb4ubljupzm4swkjm8km.
(anyscale +2.9s) status: pending
(anyscale +17.9s) status: pending
(anyscale +33.1s) status: in_progress
(anyscale +48.4s) status: in_progress
...
(anyscale +19m37.5s) status: in_progress
(anyscale +19m37.8s) Cluster environment successfully finished building.
```

Cluster env is now `cli_cluster_env_17_05_23_10_00_38_415984:1`. Put into `job_train.yaml`.

## Build cluster compute

```bash
> anyscale compute-config create compute_config.yaml -n train_config
Authenticating

Output
(anyscale +2.6s) View this cluster compute at: https://console.anyscale.com/configurations/cluster-computes/cpt_cf4uf2cr14c9cmb9mdp1z26mjy
(anyscale +2.6s) Cluster compute id: cpt_cf4uf2cr14c9cmb9mdp1z26mjy
(anyscale +2.6s) Cluster compute name: train_config
```

Cluster compute is now `train_config`. Put into `job_train.yaml`.

## Run training job

```bash
> anyscale job submit job_train.yaml --name train                   
Authenticating

Output
(anyscale +4.0s) Maximum uptime is disabled for clusters launched by this job.
(anyscale +4.0s) Job prodjob_97f11mft6wcgce77khvj29m3pd has been successfully submitted. Current state of job: PENDING.
(anyscale +4.0s) Query the status of the job with `anyscale job list --job-id prodjob_97f11mft6wcgce77khvj29m3pd`.
(anyscale +4.0s) Get the logs for the job with `anyscale job logs --job-id prodjob_97f11mft6wcgce77khvj29m3pd --follow`.
(anyscale +4.0s) View the job in the UI at https://console.anyscale.com/jobs/prodjob_97f11mft6wcgce77khvj29m3pd
```