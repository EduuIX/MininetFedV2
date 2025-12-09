import sys
from pathlib import Path
from time import sleep

from mininet.log import info, setLogLevel

from containernet.link import TCLink
from federated.net import MininetFed
from federated.node import Server, Client

volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume, "/tmp:/tmp:rw"] #volumes = [f"{Path.cwd()}:" + volume, "/tmp/.X11-unix:/tmp/.X11-unix:rw"]

# --- Experiment Configuration ---
experiment_config = {
    "ipBase": "10.0.0.0/24",
    "experiments_folder": "experiments",
    "experiment_name": "fedload_experiment",
    "date_prefix": False
}

# --- Server Arguments for FedLoad ---
# Note: The 'client_selector' and 'aggregator' are placeholders.
# The core FedLoad logic is now handled by our custom Controller.
server_args = {
    "min_trainers": 10, 
    "num_rounds": 20, 
    "stop_acc": 0.999,
    'client_selector': 'Random', # The controller will randomly select a subset of clients
    'aggregator': "FedAvg",      # The controller handles the custom aggregation
    'round_timeout': 180         # Timeout for client responses
}

# --- Client Arguments ---
# Using the modified TrainerMNIST which reports all necessary metrics
client_args = {
    "mode": 'random same_samples', 
    'num_samples': 8000,
    "trainer_class": "TrainerMNIST"
}

# --- Client Resource Limitations ---
# Using varied limitations to provide context to the bandit
bw = [10, 2, 8, 10, 5, 1, 10, 5, 10, 2]
delay = ["10ms", "100ms", "20ms", "10ms", "50ms", "150ms", "10ms", "50ms", "10ms", "80ms"]
loss = [None, 5, 1, None, 2, 8, None, 2, None, 4]
cpu_shares = [1024, 256, 768, 1024, 512, 256, 1024, 512, 1024, 256]
client_mem_lim = ["1024mb"] * 10

def topology():
    net = MininetFed(**experiment_config, controller=[], broker_mode="internal",
                     default_volumes=volumes, topology_file=sys.argv[0])

    info('*** Adding Nodes...\n')
    s1 = net.addSwitch("s1", failMode='standalone')

    srv1 = net.addHost('srv1', cls=Server, script="server/server.py",
                       args=server_args, volumes=volumes,
                       dimage='mininetfed:server')

    clients = []
    for i in range(10):
        clients.append(net.addHost(f'sta{i}', cls=Client, script="client/client.py",
                                   args=client_args, volumes=volumes,
                                   dimage='mininetfed:client',
                                   numeric_id=i, 
                                   mem_limit=client_mem_lim[i],
                                   cpu_shares=cpu_shares[i]))

    info('*** Connecting to the MininetFed Devices...\n')
    net.connectMininetFedDevices()

    info('*** Creating links with limitations...\n')
    net.addLink(srv1, s1)
    for i, client in enumerate(clients):
        net.addLink(client, s1, cls=TCLink,
                    bw=bw[i], loss=loss[i], delay=delay[i])

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