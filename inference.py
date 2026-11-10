import torch
from model import Linear_QNet

class snakeInference:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Linear_QNet(11, 256, 3).to(self.device)

        weights = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(weights)
        self.model.eval()

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)    

        if state.ndim == 1:
            state = torch.unsqueeze(state, 0)

        with torch.inference_mode():
            prediction = self.model(state)

        action = torch.argmax(prediction, dim=1).item()

        return {
            "action_index": action,
            "prediction": prediction.squeeze(0).tolist()
        }