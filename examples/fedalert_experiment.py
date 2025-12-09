# examples/fedalert_experiment.py

import sys
from pathlib import Path
from time import sleep

from mininet.log import info, setLogLevel

from containernet.link import TCLink
from federated.net import MininetFed
from federated.node import Server, Client

# --- Standard MininetFed Setup ---
volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume, "/tmp:/tmp:rw"]

# --- FedAlert Configuration (from fedalert_tf.py) ---
# This dictionary contains all the parameters for the FedAlert algorithm.
# It will be passed to both the server and the clients.
fedalert_config = {
    # --- Experiment Setup ---
    "algorithm": "FedAlert", # This tells the server to use your new ControllerFedAlert
    "drift_enabled": True,
    "mitigation_enabled": True,
    "mitigation_strategy": "hybrid",

    # --- Federated Learning Setup ---
    "num_clients": 12, # Reduced for easier testing on a local machine
    "num_rounds": 150,  # Reduced for faster experiment cycles
    "clients_per_round": 12,
    "epochs_per_client": 2,
    "batch_size": 32,
    "lr": 0.01,
    "data_distribution": "non-iid",

    # --- Drift Simulation ---
    "drift_start_round": 50, # Start drift earlier in the shorter experiment
    "drift_percentage": 0.6,
    "drift_type": "label_swap",

    # --- FedAlert Parameters ---
    "warmup_rounds": 10,
    "z_factor": 1.5,
    "alert_trigger_threshold": 0.25,
    
    # --- Mitigation Parameters ---
    "min_clean_clients": 2,
    "rollback_checkpoints": 5,
    "adaptive_lr_factor": 0.5,
    "rehabilitation_rounds": 10,
}


# --- MininetFed Network and Node Configuration ---
experiment_config = {
    "ipBase": "10.0.0.0/24",
    "experiments_folder": "experiments",
    "experiment_name": "fedalert_experiment",
    "date_prefix": False
}

# The server_args are now just the FedAlert config.
# We also include a standard 'min_trainers' and 'round_timeout'.
server_args = fedalert_config.copy()
server_args["min_trainers"] = server_args["num_clients"]
server_args["round_timeout"] = 300

# The client_args will pass the full config to each trainer
# and specify the new trainer class.
client_args = fedalert_config.copy()
client_args["trainer_class"] = "TrainerFedAlert"

# Define some basic resource limitations for the clients
client_mem_lim = ["1g"] * fedalert_config["num_clients"]
cpu_shares = [1024] * fedalert_config["num_clients"]


def topology():
    net = MininetFed(**experiment_config, controller=[], broker_mode="internal",
                     default_volumes=volumes, topology_file=sys.argv[0])

    info('*** Adding Nodes...\n')
    s1 = net.addSwitch("s1", failMode='standalone')

    srv1 = net.addHost('srv1', cls=Server, script="server/server.py",
                       args=server_args, volumes=volumes,
                       dimage='mininetfed:server')

    clients = []
    for i in range(fedalert_config["num_clients"]):
        clients.append(net.addHost(f'sta{i}', cls=Client, script="client/client.py",
                                   args=client_args, volumes=volumes,
                                   dimage='mininetfed:client', # Use the standard TF client image
                                   numeric_id=i, 
                                   mem_limit=client_mem_lim[i],
                                   cpu_shares=cpu_shares[i]))

    info('*** Connecting to the MininetFed Devices...\n')
    net.connectMininetFedDevices()

    info('*** Creating links...\n')
    net.addLink(srv1, s1)
    for client in clients:
        net.addLink(client, s1) # Simple links for now

    info('*** Starting network...\n')
    net.build()
    net.addNAT(name='nat0', linkTo='s1', ip='192.168.210.254').configDefault()
    s1.start([])

    info('*** Running FL internal devices...\n')
    net.runFlDevices()

    srv1.run(broker_addr=net.broker_addr,
             experiment_controller=net.experiment_controller)

    sleep(3)
    for client in clients:
        client.run(broker_addr=net.broker_addr,
                   experiment_controller=net.experiment_controller)

    info('*** Running Autostop...\n')
    net.wait_experiment(start_cli=False)

    info('*** Stopping network...\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology()