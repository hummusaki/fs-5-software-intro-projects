"""Learn simulated resistance, then compare three equally configured controllers"""
from pathlib import Path #locate output folder next to script
import json #save readable measurements
import matplotlib
matplotlib.use("Agg") #saves plots without requiring a desktop plotting window
import matplotlib.pyplot as plt
import numpy as np
import torch

from pid_template import (
    make_car, update, calculate_desired_acceleration,
    acceleration_to_throttle_percentage,
) #reuse car state, PID calculation, throttle conversion, and physics

MASS, MAX_FORCE = 1000.0, 5000.0
K_P, K_I, K_D = 7.0, 0.0075, 0.4
DT, SPEED_SCALE = 0.1, 15.0 #SPEED_SCALE keeps feature magnitudes around one (since we only go up to 15 m/s)
STEPS, EPOCHS = 400, 1000
OUTPUT = Path(__file__).resolve().parent / "ml_results"


def apply_acceleration(car, acceleration):
    """
    Converts accel into throttle, clips it to [-1, 1], updates car
    Inputs: car dict, requested accel
    Output: applied throttle
    """
    throttle = acceleration_to_throttle_percentage(acceleration, MASS, MAX_FORCE)
    update(car, throttle, mass=MASS, max_throttle_force=MAX_FORCE)
    return throttle


def generate_data(target_speeds):
    """
    Saves old velocity, calculates PID accel, then applies it
    Input: target speeds (each target gets a fresh car to reset integral and prev_error)
    Outputs: 2 NumPy arrays of same length with speeds and resistance
    """
    velocities, resistances = [], []
    for target in target_speeds:
        car = make_car(desired_v=target, dt=DT)
        for _ in range(STEPS):
            velocity_before = car["v"] 
            desired_accel, _ = calculate_desired_acceleration(car, K_P, K_I, K_D) #error not needed
            throttle = apply_acceleration(car, desired_accel)
            velocities.append(velocity_before) #record old velocity
            resistances.append(throttle * MAX_FORCE / MASS - car["a"])
    return np.asarray(velocities), np.asarray(resistances)


def features(velocities):
    """
    converts velocities to a 32-bit fp tensor of shape [N, 2], normalizes speed
    Input: sequence of velocities
    Output: 
    """
    v = torch.as_tensor(velocities, dtype=torch.float32) / SPEED_SCALE
    return torch.stack((v, v ** 2), dim=1) #returns tensor of shape [N, 2]


def train(velocities, resistances):
    """
    Linear layer with MSE loss, Adam optimizer, shape (2,1)
    Inputs: training speeds, resistance labels
    Outputs: model, initial/final training losses
    """
    torch.manual_seed(42) #makes experiments repeatable
    model = torch.nn.Linear(2, 1)
    inputs = features(velocities)
    labels = torch.as_tensor(resistances, dtype=torch.float32).reshape(-1, 1) #actual resistance during that step shaped into [N, 1]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #adjusts the two weights at 0.1 learning rate
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        loss = loss_fn(model(inputs), labels)
        losses.append(loss.item()) #records loss
        optimizer.zero_grad() #clears previous gradients (pytorch accumulates them)
        loss.backward() #calculate new gradients
        optimizer.step() #update parameters with new gradients
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.6f}") #progress line prints every 100 epochs
    model.eval() #evaluation mode
    with torch.no_grad():
        final_loss = loss_fn(model(inputs), labels).item() #final loss without tracking derivatives to save mem
    return model, losses[0], final_loss


def evaluate(model, velocities, resistances):
    """
    Model testing with MAE
    Inputs: trained model, held-out data
    Output: test MAE
    """
    with torch.no_grad(): #prediction without storing info needed to train the model
        predictions = model(features(velocities)).numpy().ravel() #flattened to align with the 1D resistance array
    mae = float(np.mean(np.abs(predictions - resistances))) #averages abs. errors in m/s^2
    fig, ax = plt.subplots(figsize=(8, 5)) #scatter series to show actual/predicted resistance vs speed
    ax.scatter(velocities, resistances, s=14, label="Actual", alpha=0.6)
    ax.scatter(velocities, predictions, s=10, label="Predicted", alpha=0.6)
    ax.set(xlabel="Speed (m/s)", ylabel="Resistance (m/s²)",
           title=f"Held-out targets: 7 and 12 m/s | MAE: {mae:.4f} m/s²")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout() #fits labels
    fig.savefig(OUTPUT / "resistance_predictions.png", dpi=180) #writes the png
    plt.close(fig) #release the figure
    return mae


def compare_controllers(model, fixed_resistance):
    """
    Compares plain PID, fixed correction, ML correction with 7, 12, 8 m/s targets for 20 seconds each, then plots results with MAE averages
    Inputs: model, mean training resistance
    Output: dict of controller errors
    """
    metrics = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in ("Plain PID", "Fixed correction", "ML correction"):
        car = make_car(desired_v=7.0, dt=DT) #fresh car for each model
        times, targets, velocities = [], [], []
        for target in (7.0, 12.0, 8.0):
            car["desired_v"] = target
            for _ in range(round(20.0 / DT)):
                desired_accel, _ = calculate_desired_acceleration(car, K_P, K_I, K_D)
                correction = 0.0
                if mode == "Fixed correction":
                    correction = fixed_resistance #apply fixed resistance as correction
                elif mode == "ML correction":
                    with torch.no_grad():
                        correction = max(0.0, model(features([car["v"]])).item()) #prediction is bounded below by zero
                apply_acceleration(car, desired_accel + correction)
                times.append(car["t"]) #record time
                targets.append(car["desired_v"]) #record target
                velocities.append(car["v"]) #record speed
        metrics[mode] = float(np.mean(np.abs(np.asarray(targets) - velocities))) #MAE averages abs. speed errors over 600 steps
        ax.plot(times, velocities, label=f"{mode} (MAE {metrics[mode]:.3f} m/s)") #plot trajectory with MAE
    ax.step(times, targets, where="pre", color="black", linestyle="--", label="Target")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Identical PID gains and target schedule")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "controller_comparison.png", dpi=180)
    plt.close(fig)
    return metrics


def main():
    """
    
    """
    OUTPUT.mkdir(exist_ok=True) #output dir created if needed
    train_v, train_r = generate_data((5.0, 9.0, 15.0)) #collect
    test_v, test_r = generate_data((7.0, 12.0)) #collect
    model, initial_loss, final_loss = train(train_v, train_r) #train
    test_mae = evaluate(model, test_v, test_r) #evaluate
    fixed_resistance = float(train_r.mean()) #collect (kind of)
    controller_mae = compare_controllers(model, fixed_resistance) #compare
    weights = model.weight.detach().numpy().ravel() #easier to report the weights in a numpy arr
    
    """
    Results dict. stores sample counts, losses,
    test error, controller errors, coeffs. and settings
    """
    results = {
        "training_samples": len(train_v), "test_samples": len(test_v),
        "initial_training_mse": initial_loss, "final_training_mse": final_loss,
        "test_resistance_mae_mps2": test_mae,
        "fixed_resistance_mps2": fixed_resistance,
        "controller_speed_mae_mps": controller_mae,
        "learned_equation": {"constant": model.bias.item(),
                             "speed": float(weights[0] / SPEED_SCALE),
                             "speed_squared": float(weights[1] / SPEED_SCALE ** 2)},
        "settings": {"mass": MASS, "max_force": MAX_FORCE, "dt": DT,
                     "gains": [K_P, K_I, K_D], "epochs": EPOCHS, "seed": 42},
    }
    torch.save({"model_state_dict": model.state_dict(), "speed_scale": SPEED_SCALE},
               OUTPUT / "resistance_model.pt") #writes the learned state and features scale to resistance_model.pt
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n") #saves readable measurements
    print(json.dumps(results, indent=2)) #prints measurements


#entrypoint
if __name__ == "__main__":
    main()
