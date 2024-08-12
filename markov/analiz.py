import pyemma
import cupy as cp
import argparse
import plotly.graph_objects as go
import logging
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Boltzmann sabiti ve sıcaklık (kcal/mol biriminde)
kB = 0.0019872041  # kcal/(mol·K)
T = 300  # K

def parse_arguments():
    parser = argparse.ArgumentParser(description="MSM Model Analysis with GPU acceleration.")
    parser.add_argument('msm_model', type=str, help='Path to the MSM model file')
    parser.add_argument('total_simulation_time_ns', type=float, help='Total simulation time in nanoseconds')
    parser.add_argument('n_frames', type=int, help='Total number of frames in the simulation')
    parser.add_argument('--timescale_color', type=str, default='blue', help='Color for the timescale plot')
    parser.add_argument('--energy_color', type=str, default='red', help='Color for the free energy plot')
    return parser.parse_args()

def main():
    args = parse_arguments()

    total_simulation_time_ns = args.total_simulation_time_ns
    n_frames = args.n_frames
    time_step = total_simulation_time_ns / n_frames  # Zaman adımı (ns)

    logger.info("Loading MSM model...")
    # MSM modelini yükleyin
    msm = pyemma.load(args.msm_model)
    logger.info("MSM model loaded successfully.")

    logger.info("Calculating implied timescales...")
    # Zaman ölçeklerini alın ve görselleştirin
    timescales_steps = msm.timescales()
    timescales_ns = timescales_steps * time_step  # Zaman ölçeklerini nanosecond cinsine çevirin
    fig_timescales = go.Figure()
    fig_timescales.add_trace(go.Scatter(y=timescales_ns, mode='lines+markers', marker=dict(color=args.timescale_color, size=8)))
    fig_timescales.update_layout(
        title='Implied Timescales',
        xaxis_title='Index',
        yaxis_title='Timescale (ns)',
        yaxis_type='log',
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', tickvals=[0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
                   ticktext=['0.01 ns', '0.1 ns', '1 ns', '10 ns', '100 ns', '1 µs', '10 µs', '100 µs', '1 ms']),
    )
    fig_timescales.show()
    logger.info("Implied timescales calculated and plotted successfully.")

    logger.info("Calculating state populations...")
    # Durum popülasyonlarını ve geçiş olasılıklarını inceleyin
    populations = msm.stationary_distribution
    logger.info(f"State Populations: {populations}")

    logger.info("Calculating free energy profile on GPU...")
    # Serbest enerji profilini hesaplayın ve görselleştirin
    populations_gpu = cp.array(populations)
    F = -kB * T * cp.log(populations_gpu)
    F_cpu = cp.asnumpy(F)

    # Durumların zaman aralıklarını belirleyin
    logger.info("Calculating state occurrences in the trajectory...")
    dtrajs = msm.discrete_trajectories_full
    state_durations = {state: [] for state in range(len(populations))}

    for traj in dtrajs:
        current_state = traj[0]
        start_time = 0
        for i in range(1, len(traj)):
            if traj[i] != current_state:
                if current_state in state_durations:
                    state_durations[current_state].append((start_time * time_step, i * time_step))
                current_state = traj[i]
                start_time = i
        # Son durumu ekle
        if current_state in state_durations:
            state_durations[current_state].append((start_time * time_step, len(traj) * time_step))

    # Her durumun ortalama zamanını hesapla
    state_averages = {state: np.mean([np.mean(interval) for interval in state_durations[state]]) for state in
                      state_durations if state_durations[state]}

    # Durumları ortalama zamanlarına göre sırala
    sorted_states = sorted(state_averages.keys(), key=lambda state: state_averages[state])

    # Durumları ve enerjilerini sıraya koy
    sorted_energies = [F_cpu[state] for state in sorted_states]
    sorted_labels = [
        f"State {state}\n({min(state_durations[state])[0]:.2f} ns - {max(state_durations[state])[1]:.2f} ns)" for state
        in sorted_states]

    fig_free_energy = go.Figure()
    fig_free_energy.add_trace(go.Bar(x=sorted_labels, y=sorted_energies, marker=dict(color=args.energy_color)))
    fig_free_energy.update_layout(
        title='Free Energy Profile',
        xaxis_title='State (Time Intervals)',
        yaxis_title='Free energy (kcal/mol)',
        xaxis_tickangle=-45
    )
    fig_free_energy.show()
    logger.info("Free energy profile calculated and plotted successfully.")

if __name__ == '__main__':
    main()
