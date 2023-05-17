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
> anyscale compute-config create compute_config.yaml
Authenticating

Output
(anyscale +3.3s) View this cluster compute at: https://console.anyscale.com/configurations/cluster-computes/cpt_767ylk6m215svuc8e32p23zcxa
(anyscale +3.3s) Cluster compute id: cpt_767ylk6m215svuc8e32p23zcxa
(anyscale +3.3s) Cluster compute name: cli-config-2023-05-17T10:32:42.488882
```

Cluster compute is now `cpt_767ylk6m215svuc8e32p23zcxa`. Put into `job_train.yaml`.

