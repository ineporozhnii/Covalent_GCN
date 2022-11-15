import covalent as ct
from covalent_slurm_plugin.slurm import SlurmExecutor

executor = SlurmExecutor(
    username = "",
    address = "niagara.computecanada.ca",
    ssh_key_file = "",
    remote_workdir="/scratch/user/experiment1",
    conda_env="covalent",
    options={
        "partition": "compute",
    "cpus-per-task": 8
    }
)

# Works for executor = "local", returns None for any other executor
@ct.electron(executor=executor)
def compute_pi(n):
    # Leibniz formula for π
    return 4 * sum(1.0/(2*i + 1)*(-1)**i for i in range(n))

@ct.lattice
def workflow(n):
    return compute_pi(n)


dispatch_id = ct.dispatch(workflow)(1000)
result = ct.get_result(dispatch_id=dispatch_id, wait=True)
print(result.result)