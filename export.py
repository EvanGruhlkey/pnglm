from engine.loader import load_weights
from format.serialize import save_model

model = load_weights()

save_model(model, "model.bin")