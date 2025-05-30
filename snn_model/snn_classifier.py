from brian2 import *

def run_snn(rates, duration=100*ms):
    start_scope()

    N = len(rates)
    G = PoissonGroup(N, rates)
    M = SpikeMonitor(G)

    run(duration)

    spikes = M.count
    return spikes
