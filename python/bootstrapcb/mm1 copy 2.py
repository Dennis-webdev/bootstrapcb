import random
import copy
import numpy as np
import matplotlib.pyplot as plt

LAMBDAS = np.linspace(2,8,num=5)
MU = 10

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
        self.time_series = {0: 0}
        self.job_count_hist = {}
        self.jobs = {} 
        self.job_list = []
        self.processing_job = None
        self.queue_list = []
        self.finished_jobs = []
        
    def run(self, simulation_time):
        print("\nTime: 0 sec, Simulation starts for lambda=" + str(self.arrival_rate))
        
        job_id = 1
        current_time = 0
        while current_time < simulation_time:
            new_job = Job(job_id)
            inter_arrival_time = np.random.exponential(1/self.arrival_rate)
            new_job.arrival_time = current_time + inter_arrival_time
            new_job.service_time = np.random.exponential(1/self.service_rate)
            self.jobs[job_id] = new_job         
            self.job_list.append(new_job)       
            job_id += 1
            current_time = new_job.arrival_time

        sim_time = simulation_time/1000
        current_time = 0
        while sim_time <= simulation_time:
            while True:
                if  (   len(self.job_list) > 0 and 
                        self.processing_job and
                        self.processing_job.service_end_time <= self.job_list[0].arrival_time ):
                    if self.processing_job.service_end_time > sim_time: break
                    finished_job = self.processing_job 
                    self.processing_job = None   
                    self.finished_jobs.append(finished_job)
                    current_time = finished_job.service_end_time
                    self.time_series[current_time] = len(self.queue_list)
                    continue
                if  (   len(self.job_list) > 0 and 
                        self.processing_job and
                        self.processing_job.service_end_time > self.job_list[0].arrival_time ):
                    if self.job_list[0].arrival_time > sim_time: break
                    new_job = self.job_list[0]
                    self.job_list.remove(new_job)
                    self.queue_list.append(new_job)
                    current_time = new_job.arrival_time
                    self.time_series[current_time] = len(self.queue_list) + 1
                    continue
                if not self.processing_job:
                    if  (   len(self.job_list) > 0 and 
                            len(self.queue_list) == 0 ):
                        if self.job_list[0].arrival_time > sim_time: break
                        new_job = self.job_list[0]
                        self.job_list.remove(new_job)
                        self.queue_list.append(new_job)
                        current_time = new_job.arrival_time
                        self.time_series[current_time] = len(self.queue_list) 
                    self.processing_job = self.queue_list[0]
                    self.processing_job.service_start_time = current_time
                    self.processing_job.service_end_time = current_time + self.processing_job.service_time
                    self.queue_list.remove(self.processing_job)
                    continue
                break

            self.job_count_hist[current_time] = len(self.queue_list) + (1 if self.processing_job else 0) 
            sim_time += simulation_time/1000

        print("Total jobs: " + str(len(self.jobs)))
        return self.time_series, self.job_count_hist, self.jobs
        
# def plot_simulation_jobs_vs_t(jobs, arrival_rate, sumarize):

if __name__ == '__main__':
    results = {}
    for arrival_rate in LAMBDAS:   
        system = System(arrival_rate, MU)
        time_series, job_count_hist, jobs = system.run(TOTAL_SIMULATION_TIME)

        # x = np.array([x for x in result])
        # y = np.array([result[x] for x in result])
        # plt.step(x,y, 'g^--', where='post')
        # plt.show()

        job_count_hist = [job_count_hist[c] for c in job_count_hist]   
        plt.hist(job_count_hist, density=True)
        rho = arrival_rate/MU
        testx = np.linspace(min(job_count_hist), max(job_count_hist), num=50)
        testy = [(1-rho)*rho**n for n in testx]
        plt.plot(testx, testy)
        plt.show()
        for _ in range(10):
            choice = np.random.choice(job_count_hist,size=3000)
            test_hist = [c for c in choice]   
            plt.hist(test_hist, density=True)
            plt.plot(testx, testy)
            plt.show()

        average = 0
        first_t = None
        last_t = None
        for t in time_series:
            if last_t == None:
                first_t = t
                last_t = t
            else: 
                average += time_series[last_t] * (t - last_t)
                last_t = t 
        average /= (last_t - first_t)

        average = np.average([c for c in job_count_hist] )

        rho = arrival_rate/MU
        results[arrival_rate] = [ average,
                                  rho / (1 - rho) ] 

    lamdas = [lamda for lamda in results]
    the_simulation_data = [lamdas, [results[lamda][0] for lamda in lamdas]]
    the_theoretical_data = [lamdas, [results[lamda][1] for lamda in lamdas]]

    plt.figure(" Comparison")
    axis = plt.subplot()

    plt.plot(the_simulation_data[0], the_simulation_data[1], 'b--')
    plt.plot(the_simulation_data[0], the_simulation_data[1], 'bs', label='Simulation average')

    plt.plot(the_theoretical_data[0], the_theoretical_data[1], 'g--')
    plt.plot(the_theoretical_data[0], the_theoretical_data[1], 'go', label='Theoretical E[T]')

    axis.set_xlabel('Rho Value')
    axis.set_ylabel('Average jobs in queue')
    axis.legend()
    axis.set_title("Simulation vs. steady-state: Jobs in queue on M/M/1 " + ", Simulation time: " + str(
        TOTAL_SIMULATION_TIME) + "secs")
    plt.show()
    plt.close()