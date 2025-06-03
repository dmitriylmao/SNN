import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_snn_steps, output_size=2, beta=0.95):
        super().__init__()
        self.num_snn_steps = num_snn_steps
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beta = beta

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=surrogate.atan(), output=True)

    def forward(self, x_sequence):
        batch_size = x_sequence.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        out_potentials_over_time = []

        for step in range(self.num_snn_steps):
            current_input = x_sequence[:, step, :]
            
            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            out_potentials_over_time.append(cur2)

        final_out_potentials = torch.stack(out_potentials_over_time, dim=1).sum(dim=1)

        return final_out_potentials

if __name__ == '__main__':
    batch_size = 4
    sequence_length = 10
    num_features = 5

    test_input_sequence = torch.randn(batch_size, sequence_length, num_features)

    snn_model = SNNClassifier(input_size=num_features, hidden_size=32, num_snn_steps=sequence_length, output_size=2)

    logits = snn_model(test_input_sequence)

    print("Форма входных данных:", test_input_sequence.shape)
    print("Форма выходных логитов:", logits.shape)