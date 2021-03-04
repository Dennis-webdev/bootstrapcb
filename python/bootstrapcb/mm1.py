import random
import copy
import numpy as np
import matplotlib.pyplot as plt

LAMBDAS = np.linspace(0.2,0.8,num=5)
MU = 1

TOTAL_SIMULATION_TIME = 10000  # time for running simulation

# class to create an object that represent a single job
class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.arrival_time = 0
        self.service_time = 0
        self.service_start_time = 0
        self.service_end_time = 0
        
class System:
    def __init__(self, arrival_rate, service_rate):
        self.arrival_rate = arrival_rate 
        self.service_rate = service_rate 
        self.current_time = 0  
        self.queue_list = []
        self.time_series = {self.current_time: 0}
        self.processing_job = None
        self.finished_jobs = []
        
    def add_to_queue_and_process(self, new_job):
        current_time = self.current_time

        inter_arrival_time = np.random.exponential(1/self.arrival_rate)
        new_job.arrival_time = current_time + inter_arrival_time
        new_job.service_time = np.random.exponential(1/self.service_rate)

        while True:
            if self.processing_job == None:
                if len(self.queue_list) > 0:
                    self.processing_job = self.queue_list[0]
                    self.processing_job.service_start_time = current_time
                    self.processing_job.service_end_time = current_time + self.processing_job.service_time
                    self.queue_list.remove(self.processing_job)
                else:
                    break
            if self.processing_job.service_end_time < new_job.arrival_time:
                current_time = self.processing_job.service_end_time
                self.finished_jobs.append(self.processing_job)
                self.processing_job = None     
                self.time_series[current_time] = len(self.queue_list)    
            else:
                break    
                
        current_time = new_job.arrival_time
        self.queue_list.append(new_job)
        self.time_series[current_time] = len(self.queue_list) + (1 if self.processing_job else 0)
        self.current_time = current_time

class Simulator:
    def __init__(self, arrival_rate, service_rate):
        self.system = System(arrival_rate, service_rate)
        
    def run(self, simulation_time):
        print("\nTime: 0 sec, Simulation starts for λ=" + str(self.system.arrival_rate))
        this_jobs = {}  # map of id:job
        time_series = {}
        job_id = 1

        while self.system.current_time < simulation_time:
            new_job = Job(job_id)
            this_jobs[job_id] = new_job         
            self.system.add_to_queue_and_process(new_job)
            job_id += 1

        print("Total jobs: " + str(len(this_jobs)))
        return self.system.time_series, this_jobs
        
# def plot_simulation_jobs_vs_t(jobs, arrival_rate, sumarize):

if __name__ == '__main__':
    results = {}
    average = {}
    for arrival_rate in LAMBDAS:   
        simulator = Simulator(arrival_rate, MU)
        result, jobs = simulator.run(TOTAL_SIMULATION_TIME)

        # x = np.array([x for x in result])
        # y = np.array([result[x] for x in result])
        # plt.step(x,y, 'g^--', where='post')
        # plt.show()        

        average_jobs = 0
        for item in result:
            if item > 0: 
                average_jobs += result[last_item] * (item - last_item)
            last_item = item 
        average_jobs /= last_item
        results[arrival_rate] = [ average_jobs,
                                  arrival_rate / (MU - arrival_rate) ] 

    lamdas = [lamda for lamda in results]
    the_simulation_data = [lamdas, [results[lamda][0] for lamda in lamdas]]
    the_theoretical_data = [lamdas, [results[lamda][1] for lamda in lamdas]]

    plt.figure(" Comparison")
    axis = plt.subplot()

    plt.plot(the_simulation_data[0], the_simulation_data[1], 'b--')
    plt.plot(the_simulation_data[0], the_simulation_data[1], 'bs', label='Jobs in queue')

    plt.plot(the_theoretical_data[0], the_theoretical_data[1], 'g--')
    plt.plot(the_theoretical_data[0], the_theoretical_data[1], 'go', label='Theoretical E[T]')

    axis.set_xlabel('Lambda Value')
    axis.set_ylabel('Jobs in queue')
    axis.legend()
    axis.set_title("Simulation vs. steady-state: Jobs in queue on M/M/1 " + ", Simulation time: " + str(
        TOTAL_SIMULATION_TIME) + "secs")
    plt.show()
    plt.close()