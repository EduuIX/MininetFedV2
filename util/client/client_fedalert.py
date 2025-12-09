# util/client/client.py

import paho.mqtt.client as mqtt
import numpy as np
import json
import time
import sys
import importlib

def criar_objeto(pacote, nome_classe, **atributos):
    """
    Dynamically creates an object by importing the 'trainer' package 
    and getting the class by its name. This relies on trainer/__init__.py.
    """
    try:
        # Import the trainer package. __init__.py makes the classes available.
        modulo = importlib.import_module(pacote) 
        # Get the specific class (e.g., TrainerFedAlert) from the package.
        classe = getattr(modulo, nome_classe)
        return classe(**atributos)
    except Exception as e:
        print(f"Error creating trainer: {e}", file=sys.stderr)
        return None

def default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class color:
    GREEN = '\033[92m'
    BOLD_END = '\033[0m'

# --- Main Client Logic ---
if len(sys.argv) < 5:
    print("Usage: python client.py <broker> <name> <id> <args_json>")
    exit()

BROKER_ADDR, CLIENT_NAME, CLIENT_ID, CLIENT_ARGS_JSON = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
CLIENT_ARGS = json.loads(CLIENT_ARGS_JSON)
TRAINER_CLASS = CLIENT_ARGS.get("trainer_class")

# The first argument to criar_objeto is the package name "trainer"
trainer = criar_objeto("trainer", TRAINER_CLASS, id=CLIENT_ID, name=CLIENT_NAME, args=CLIENT_ARGS)
if trainer is None: exit()

def on_connect(client, userdata, flags, rc):
    client.subscribe(f'minifed/clients/{CLIENT_NAME}')

def on_message(client, userdata, message):
    try:
        msg = json.loads(message.payload.decode("utf-8"))
        action = msg.get('action')
        
        if action == "set_config":
            trainer.set_args(msg['config'])
            print("Configuration received from server.")
            
        elif action == "train_round":
            print("Received training command...")
            trainer.update_weights([np.asarray(w) for w in msg['weights']])
            trainer.train_model()
            response = {
                "weights": trainer.get_weights(),
                "metrics": trainer.all_metrics(),
                "num_samples": trainer.get_num_samples()
            }
            client.publish('minifed/client_responses', json.dumps({'id': CLIENT_NAME, 'response': response}, default=default))
            
            # --- FINALIZED: Print metrics to the local terminal ---
            metrics = response['metrics']
            acc = metrics.get('accuracy', 0)
            loss_imp = metrics.get('loss_improvement', 0)
            is_alert = metrics.get('is_alert', False)
            print(color.GREEN + f"Training complete. Accuracy: {acc*100:.2f}% | Loss Imp: {loss_imp:.4f} | Alert: {is_alert}" + color.BOLD_END)

        elif action == "set_local_threshold":
            trainer.set_local_threshold()
            
        elif action == "apply_drift":
            trainer.apply_drift()
            client.publish('minifed/client_responses', json.dumps({'id': CLIENT_NAME, 'response': {'status': 'drift_applied'}}, default=default))

        elif action == "update_learning_rate":
            trainer.adaptive_lr *= msg.get('factor', 0.5)
            print(f"Adaptive LR updated to {trainer.adaptive_lr}")

        elif action == "stop":
            print("Received stop command. Shutting down.")
            trainer.set_stop_true()

    except Exception as e:
        print(f"Error processing message: {e}", file=sys.stderr)

client = mqtt.Client(CLIENT_NAME)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_ADDR, 1883)
client.loop_start()

initial_metrics = trainer.all_metrics()
initial_weights = trainer.get_weights()
reg_payload = { 'id': CLIENT_NAME, 'metrics': initial_metrics, 'initial_weights': initial_weights }
client.publish('minifed/registerQueue', json.dumps(reg_payload, default=default))
print(f"Client {CLIENT_NAME} registered.")

while not trainer.get_stop_flag():
    time.sleep(1)

client.loop_stop()
print(f"Client {CLIENT_NAME} has stopped.")