# util/server/server.py

import paho.mqtt.client as mqtt
import json
import time
import numpy as np
import sys
import logging
import os
import copy

from controller_fedalert import ControllerFedAlert

def default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

FORMAT = "%(asctime)s - %(infotype)-6s - %(levelname)s - %(message)s"
class color:
    YELLOW = '\033[93m'; RED = '\033[91m'; BOLD_START = '\033[1m'; RESET = "\x1B[0m"; GREEN = '\033[92m'

def server():
    os.umask(0o000)
    if len(sys.argv) < 4:
        print("correct use: python server.py <broker_address> <log_file> <args_json>")
        exit()

    server_args = json.loads(sys.argv[3])
    broker_addr, log_file = sys.argv[1], sys.argv[2]
    
    logging.basicConfig(level=logging.INFO, filename=log_file, format=FORMAT, filemode="w")
    logger = logging.getLogger(__name__)
    execType = {"infotype": "EXECUT"}
    
    controller = ControllerFedAlert(server_args)

    def on_connect(client, userdata, flags, rc):
        client.subscribe('minifed/registerQueue')
        client.subscribe('minifed/client_responses')

    def on_message_register(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.add_trainer(m["id"])
        controller.update_metrics(m["id"], m['metrics'])
        if 'initial_weights' in m:
            controller.set_initial_global_model(m['initial_weights'])
        logger.info(f'Trainer {m["id"]} joined the pool.', extra=execType)
        print(f'Trainer {m["id"]} joined the pool.')
        client.publish(f'minifed/clients/{m["id"]}', json.dumps({"action": "set_config", "config": server_args}))

    def on_client_response(client, userdata, message):
        m = json.loads(message.payload.decode("utf-8"))
        controller.add_client_response(m['id'], m['response'])

    client = mqtt.Client('server')
    client.on_connect = on_connect
    client.message_callback_add('minifed/client_responses', on_client_response)
    client.message_callback_add('minifed/registerQueue', on_message_register)
    client.connect(broker_addr, 1883)
    client.loop_start()

    logger.info("FedAlert Server Starting...", extra=execType)
    print(color.BOLD_START + 'FedAlert Server Starting...' + color.RESET)

    while controller.state != "FINISHED":
        action, target_clients = controller.next_action()
        if action is None:
            time.sleep(1) 
            continue
        
        # Log phase changes and special actions
        if action.get("phase"):
            logger.info(f"--- Phase: {action['phase']} ---", extra=execType)
            print(color.BOLD_START + f"\n--- Phase: {action['phase']} ---" + color.RESET)
            if action['phase'] == "MAIN TRAINING, DETECTION & MITIGATION":
                # This is a one-way command, no response expected
                for client_id in target_clients:
                    client.publish(f'minifed/clients/{client_id}', json.dumps(action, default=default))
            continue
        elif action.get("action") == "apply_drift":
             logger.info(f"!!! Introducing '{server_args['drift_type']}' drift to clients: {target_clients} !!!", extra=execType)
             print(color.RED + f"!!! Introducing '{server_args['drift_type']}' drift to clients: {target_clients} !!!" + color.RESET)
             for client_id in target_clients:
                 client.publish(f'minifed/clients/{client_id}', json.dumps(action, default=default))
             # We need to wait for confirmation from these clients
        
        logger.info(f"Round {controller.current_round} | State: {controller.state} | Action: {action['action']}", extra=execType)
        print(color.BOLD_START + f"\nRound {controller.current_round} | State: {controller.state} | Action: {action['action']}" + color.RESET)
        
        controller.client_responses.clear()
        
        if 'weights' in action or action['action'] in ['train_round']:
             action['weights'] = [w.tolist() for w in controller.global_model_weights]
        
        for client_id in target_clients:
            client.publish(f'minifed/clients/{client_id}', json.dumps(action, default=default))
        
        if action['action'] == 'stop': break

        if action['action'] not in ['set_local_threshold', 'stop', 'update_learning_rate']:
            timeout = server_args.get("round_timeout", 300)
            start_time = time.time()
            while len(controller.client_responses) < len(target_clients):
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout! Received {len(controller.client_responses)}/{len(target_clients)} responses.", extra=execType)
                    print(color.YELLOW + f"Timeout! Received {len(controller.client_responses)}/{len(target_clients)} responses." + color.RESET)
                    break
                time.sleep(1)

        old_weights = copy.deepcopy(controller.global_model_weights)
        mitigation_action = controller.process_responses(old_weights, logger) # Pass logger here
        
        if mitigation_action and mitigation_action['action'] == 'update_learning_rate':
             for client_id in mitigation_action['client_ids']:
                 client.publish(f'minifed/clients/{client_id}', json.dumps(mitigation_action))

    controller.final_analysis(logger)
    
    logger.info("Experiment Finished. Telling all nodes to stop.", extra=execType)
    print(color.RED + "Experiment Finished. Telling all nodes to stop." + color.RESET)
    for client_id in controller.trainer_list:
        client.publish(f'minifed/clients/{client_id}', json.dumps({"action": "stop"}))
    
    client.publish('minifed/stopQueue', json.dumps({'stop': True}))
    
    time.sleep(2)
    client.loop_stop()

if __name__ == "__main__":
    server()