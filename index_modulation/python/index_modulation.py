import simulation
import time

r = 3
m = simulation.Simulation(r_tx = 0.5, plot=True)

start = time.time()
output, output_coordinates = m.start_simulation()

print("total time = " + str(time.time() - start))
summ  = 0
for i in output_coordinates:
    summ = summ + i.shape[0]
    
