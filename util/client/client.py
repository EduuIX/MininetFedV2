# util/client/client.py

import importlib
import paho.mqtt.client as mqtt
import numpy as np
import json
import time
import sys
try:
    import torch
except:
    pass

def criar_objeto(pacote, nome_classe, **atributos):
    try:
        modulo = importlib.import_module(f"{pacote}")
        classe = getattr(modulo, nome_classe)
        return classe(**atributos)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return None

# --- Argument Parsing ---
n = len(sys.argv)
if n != 4 and n != 5:
    print("correct use: python client.py <broker_address> <name> <id> [client_instanciation_args].")
    exit()

BROKER_ADDR = sys.argv[1]
CLIENT_NAME = sys.argv[2]
CLIENT_ID = int(sys.argv[3])
CLIENT_INSTANTIATION_ARGS = {}
if len(sys.argv) == 5 and (sys.argv[4] is not None):
    CLIENT_INSTANTIATION_ARGS = json.loads(sys.argv[4])

trainer_class = CLIENT_INSTANTIATION_ARGS.get("trainer_class")
if trainer_class is None:
    trainer_class = "TrainerMNIST"

selected = False

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    elif 'torch' in sys.modules and type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    else:
        try:
            from Pyfhel import PyCtxt
            if isinstance(obj, PyCtxt):
                return obj.to_bytes().decode('cp437')
        except:
            pass
    raise TypeError('Tipo não pode ser serializado:', type(obj))

def has_method(o, name):
    return callable(getattr(o, name, None))

class color:
    BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BOLD_START = '\033[1m'; BOLD_END = '\033[0m'; RESET = "\x1B[0m"

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    subscribe_queues = [
        'minifed/selectionQueue', 'minifed/posAggQueue', 'minifed/stopQueue', 
        'minifed/serverArgs', 'minifed/submodelQueue' # No longer needs initialModelQueue
    ]
    for s in subscribe_queues:
        client.subscribe(s)

def on_server_args(client, userdata, message):
    msg = json.loads(message.payload.decode("utf-8"))
    if msg['id'] == CLIENT_NAME:
        if msg['args'] is not None:
            trainer.set_args(msg['args'])
        client.publish('minifed/ready', json.dumps({"id": CLIENT_NAME}, default=default))

def on_message_selection(client, userdata, message):
    global selected
    msg = json.loads(message.payload.decode("utf-8"))
    if msg['id'] == CLIENT_NAME:
        selected = bool(msg['selected'])
        if selected:
            print(color.BOLD_START + 'Selected for round, waiting for submodel.' + color.BOLD_END)
        else:
            # When not selected, still need to send metrics back later.
            client.publish('minifed/metricsQueue', json.dumps({'id': CLIENT_NAME, "metrics": trainer.all_metrics()}, default=default))
            print(color.BOLD_START + 'Not selected. Sent metrics for next round.' + color.BOLD_END)


# --- MODIFIED: Renamed from on_submodel_receive for clarity ---
def on_train_request(client, userdata, message):
    msg = json.loads(message.payload.decode("utf-8"))
    if msg['id'] == CLIENT_NAME and msg['action'] == 'train_submodel':
        print(f"Received submodel with pruning rate {msg['pruning_rate']}. Starting training.")
        
        submodel_weights = [np.asarray(w, dtype=np.float32) for w in msg["weights"]]
        trainer.update_weights(submodel_weights)
        
        trainer.train_model()
        
        updated_weights = trainer.get_weights()
        metrics = trainer.all_metrics()
        
        response = {
            'id': CLIENT_NAME,
            'weights': [w.tolist() for w in updated_weights],
            'num_samples': trainer.get_num_samples(),
            'metrics': metrics
        }
        client.publish('minifed/preAggQueue', json.dumps(response, default=default))
        print("Finished training and sent updated submodel and metrics.")

def on_message_agg(client, userdata, message):
    # This message now primarily serves as a signal to non-selected clients to send their metrics.
    # The logic for non-selected clients is handled in on_message_selection for simplicity.
    print("Round complete notification received.")

def on_message_stop(client, userdata, message):
    print(color.RED + 'Received message to stop!' + color.RESET)
    trainer.set_stop_true()
    exit()

# --- Main Client Logic ---
trainer = criar_objeto("trainer", trainer_class, id=CLIENT_ID,
                       name=CLIENT_NAME, args=CLIENT_INSTANTIATION_ARGS)
if trainer is None:
    print(color.RED + "Failed to create trainer object. Exiting." + color.RESET)
    exit()

client = mqtt.Client(str(CLIENT_NAME))
client.connect(BROKER_ADDR, keepalive=0)
client.on_connect = on_connect
client.message_callback_add('minifed/selectionQueue', on_message_selection)
client.message_callback_add('minifed/posAggQueue', on_message_agg)
client.message_callback_add('minifed/stopQueue', on_message_stop)
client.message_callback_add('minifed/serverArgs', on_server_args)
client.message_callback_add('minifed/submodelQueue', on_train_request)

client.loop_start()

# --- MODIFIED: Send initial weights on first registration ---
initial_metrics = trainer.all_metrics()
initial_weights = trainer.get_weights()
registration_payload = {
    'id': CLIENT_NAME,
    'metrics': initial_metrics,
    'initial_weights': [w.tolist() for w in initial_weights]
}
client.publish('minifed/registerQueue', json.dumps(registration_payload, default=default))
print(color.BOLD_START + f'Trainer {CLIENT_NAME} connected and sent initial model!' + color.BOLD_END)

while not trainer.get_stop_flag():
    time.sleep(1)

client.loop_stop()