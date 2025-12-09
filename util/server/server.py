# util/server/server.py

import paho.mqtt.client as mqtt
from controller import Controller
import json
import time
import numpy as np
import sys
import logging
import os

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray): return obj.tolist()
        else: return obj.item()
    else:
        try:
            from Pyfhel import PyCtxt
            if isinstance(obj, PyCtxt): return obj.to_bytes().decode('cp437')
        except: pass
    raise TypeError('Tipo não pode ser serializado:', type(obj))

FORMAT = "%(asctime)s - %(infotype)-6s - %(levelname)s - %(message)s"
class color:
    BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BOLD_START = '\033[1m'; BOLD_END = '\033[0m'; RESET = "\x1B[0m"

def server():
    os.umask(0o000)
    n = len(sys.argv)
    if n < 4:
        logging.critical("incorrect use of server.py arguments")
        print("correct use: python server.py <broker_address> <arquivo.log> <args>.")
        exit()

    server_args = json.loads(sys.argv[3])
    broker_addr = sys.argv[1]; log_file = sys.argv[2]
    min_trainers = server_args["min_trainers"]; client_selector = server_args["client_selector"]
    aggregator = server_args["aggregator"]; nun_rounds = server_args["num_rounds"]
    stop_acc = server_args["stop_acc"]; client_args = server_args.get("client")
    round_timeout = server_args.get("round_timeout", 300)

    logging.basicConfig(level=logging.INFO, filename=log_file, format=FORMAT, filemode="w")
    metricType = {"infotype": "METRIC"}; executionType = {"infotype": "EXECUT"}
    logger = logging.getLogger(__name__)

    def on_connect(client, userdata, flags, rc):
        subscribe_queues = ['minifed/registerQueue', 'minifed/preAggQueue', 'minifed/metricsQueue', 'minifed/ready']
        for s in subscribe_queues: client.subscribe(s)

    def on_message_ready(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.add_trainer(m["id"])

    # --- MODIFIED: Capture initial weights on registration ---
    def on_message_register(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.update_metrics(m["id"], m['metrics'])
        if 'initial_weights' in m:
            controller.set_initial_global_model(m['initial_weights'])
        logger.info(f'trainer number {m["id"]} just joined the pool', extra=executionType)
        print(f'trainer number {m["id"]} just joined the pool')
        client.publish('minifed/serverArgs', json.dumps({"id": m["id"], "args": client_args}))

    def on_message_agg(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.add_client_training_response(m['id'], m)
        controller.update_num_responses()
        logger.info(f'received weights from trainer {m["id"]}!', extra=executionType)
        print(f'received weights from trainer {m["id"]}!')

    def on_message_metrics(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.add_accuracy(m['metrics']['accuracy'])
        controller.update_metrics(m["id"], m['metrics'])
        m["metrics"]["client_name"] = m["id"]
        logger.info(f'{json.dumps(m["metrics"])}', extra=metricType)
        controller.update_num_responses()

    def wait_for_responses(expected_responses, timeout):
        start_time = time.time()
        while controller.get_num_responses() < expected_responses:
            if time.time() - start_time > timeout:
                logger.warning(f'Timeout! Proceeding with {controller.get_num_responses()}/{expected_responses} responses.', extra=executionType)
                print(color.YELLOW + f'Timeout! Proceeding with {controller.get_num_responses()}/{expected_responses} responses.' + color.RESET)
                break
            time.sleep(1)

    controller = Controller(min_trainers=min_trainers, num_rounds=nun_rounds, client_selector=client_selector, aggregator=aggregator)
    client = mqtt.Client('server')
    client.connect(broker_addr, bind_port=1883)
    client.on_connect = on_connect
    client.message_callback_add('minifed/registerQueue', on_message_register)
    client.message_callback_add('minifed/preAggQueue', on_message_agg)
    client.message_callback_add('minifed/metricsQueue', on_message_metrics)
    client.message_callback_add('minifed/ready', on_message_ready)
    client.loop_start()
    logger.info('starting server...', extra=executionType)
    print(color.BOLD_START + 'starting server...' + color.BOLD_END)
    client.publish('minifed/autoWaitContinue', json.dumps({'continue': True}))

    # --- MODIFIED: Wait for initial model before starting rounds ---
    print("Waiting for clients to register and provide an initial model...")
    while controller.get_num_trainers() < min_trainers or controller.global_model_weights is None:
        time.sleep(1)
    print("Initial model acquired. Starting training rounds.")

    while controller.get_current_round() != nun_rounds:
        controller.update_current_round()
        logger.info(f'round: {controller.get_current_round()}', extra=metricType)
        print(color.RESET + '\n' + color.BOLD_START + f'starting round {controller.get_current_round()}' + color.BOLD_END)
        
        select_trainers = controller.select_trainers_for_round()
        selected_qtd = len(select_trainers)
        controller.reset_num_responses()

        logger.info(f"n_selected: {len(select_trainers)}", extra=metricType)
        logger.info(f"{json.dumps({'selected_trainers': select_trainers})}", extra=metricType)
        
        # --- MODIFIED: FedLoad - Prune and send submodels ---
        for t in controller.get_trainer_list():
            if t in select_trainers:
                client_context = controller.metrics.get(t, {})
                pruning_rate = controller.bandit.choose_pruning_rate(client_context)
                controller.client_pruning_rates[t] = pruning_rate
                pruned_model = controller.prune_model(controller.global_model_weights, pruning_rate)
                
                model_payload = {'id': t, 'action': 'train_submodel', 'weights': [w.tolist() for w in pruned_model], 'pruning_rate': pruning_rate}
                client.publish('minifed/submodelQueue', json.dumps(model_payload))
                print(f'Selected trainer {t} and sent submodel (pruning rate: {pruning_rate})')
            else:
                 m = json.dumps({'id': t, 'selected': False}).replace(' ', '')
                 client.publish('minifed/selectionQueue', m)

        print(f"Waiting for weights from {selected_qtd} clients (timeout: {round_timeout}s)...")
        wait_for_responses(selected_qtd, round_timeout)
        
        agg_weights = controller.agg_weights()
        
        # After aggregation, the server needs to notify non-training clients to send their metrics for the next round.
        # The posAggQueue is a good signal for this.
        client.publish('minifed/posAggQueue', json.dumps({'status': 'round_complete'}))
        
        print(f"Waiting for metrics from all {controller.get_num_trainers()} clients (timeout: {round_timeout}s)...")
        wait_for_responses(controller.get_num_trainers(), round_timeout)
        controller.reset_num_responses()
        mean_acc = controller.get_mean_acc()
        logger.info(f'mean_accuracy: {mean_acc}\n', extra=metricType)
        print(color.GREEN + f'mean accuracy on round {controller.get_current_round()} was {mean_acc}\n' + color.RESET)

        if mean_acc >= stop_acc:
            logger.info('stop_condition: accuracy', extra=metricType)
            print(color.RED + f'accuracy threshold met! stopping the training!')
            m = json.dumps({'stop': True})
            client.publish('minifed/stopQueue', m)
            time.sleep(1)
            exit()
        controller.reset_acc_list()

    logger.info('stop_condition: rounds', extra=metricType)
    print(color.RED + f'rounds threshold met! stopping the training!' + color.RESET)
    client.publish('minifed/stopQueue', json.dumps({'stop': True}))
    client.loop_stop()

if __name__ == "__main__":
    server()