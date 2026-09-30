import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 0.1
K_I = 0.1
K_D = 0.1
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

#WRITE CODE HERE
def run(STEPS):
    velocities = []
    errors = []
    times = []
    for x in range(STEPS):
        desired_accel, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
        throttle = acceleration_to_throttle_percentage(desired_accel)
        update(car, throttle)
        velocities.append(car["v"])
        errors.append(error)
        times.append(car["t"])
        
    # figure with 2 plots, left and right
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    # velocities plot
    ax1.plot(times, velocities)
    ax1.set_title("Velocities plot")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocitiy")
    ax1.grid(True)
    
    # errors plot
    ax2.plot(times, errors)
    ax2.set_title("Errors plot")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Errors")
    ax2.grid(True)
    
    plt.tight_layout() #prevent the x axis from overlapping
    plt.show()
    
run(STEPS)