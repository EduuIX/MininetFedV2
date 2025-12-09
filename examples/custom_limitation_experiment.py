import sys
from pathlib import Path
from time import sleep

from mininet.log import info, setLogLevel

from containernet.link import TCLink
from federated.net import MininetFed
from federated.node import Server, Client

volume = "/flw"
volumes = [f"{Path.cwd()}:" + volume, "/tmp/.X11-unix:/tmp/.X11-unix:rw"]

# --- Experiment Configuration ---
experiment_config = {
    "ipBase": "10.0.0.0/24",
    "experiments_folder": "experiments",
    "experiment_name": "custom_limitations",
    "date_prefix": False
}

server_args = {"min_trainers": 10, "num_rounds": 20, "stop_acc": 0.999,
               'client_selector': 'SCOPEFL', 'aggregator': "FedAvg", 'round_timeout': 60}
               
client_args = {"mode": 'random', 
               "trainer_class": "TrainerCifar"}

# --- Client Limitation Lists (Now for 10 clients) ---
# Bandwidth in Mbps for each client's link
bw = [10, 5, 1, 10, 2, 8, 10, 5, 1, 10]
# Delay for each client's link
delay = ["10ms", "50ms", "100ms", "10ms", "80ms", "20ms", "10ms", "50ms", "100ms", "10ms"]
# Packet loss percentage for each client's link
loss = [None, 2, 5, None, 3, 1, None, 2, 5, None]
# Relative CPU shares for each client container
cpu_shares = [1024, 512, 256, 1024, 512, 768, 1024, 512, 256, 768]
# Memory limit for each client container
client_mem_lim = ["1512m"]*10 #["512m", "512m", "512m", "512m", "512m", "512m", "512m", "512m", "512m", "512m"]


def topology():
    net = MininetFed(**experiment_config, controller=[], broker_mode="internal",
                     default_volumes=volumes, topology_file=sys.argv[0])

    info('*** Adding Nodes...\n')
    s1 = net.addSwitch("s1", failMode='standalone')

    srv1 = net.addHost('srv1', cls=Server, script="server/server.py",
                       args=server_args, volumes=volumes,
                       dimage='mininetfed:server')

    clients = []
    # --- Increased the number of clients to 10 ---
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
        # Apply the limitations for each client from the lists
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