import sys

from anyscale import AnyscaleSDK
sdk = AnyscaleSDK()

res = sdk.search_cluster_environments({
    "name": {"equals": sys.argv[1]}
})
apt_id = res.results[0].id
res = sdk.list_cluster_environment_builds(apt_id)
bld_id = res.results[-1].id
print(bld_id)
