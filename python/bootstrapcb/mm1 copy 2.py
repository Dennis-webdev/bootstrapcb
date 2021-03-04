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
        self.status = 0  # 0 for created, 1 for queued, 2 for processing, 3 for completed

class System:
    def __init__(self, arrival_rate, service_rate):
        self.arrival_rate = arrival_rate 
        self.service_rate = service_rate 
        self.jobs = {} 
        self.time_series = {0: 0}
        self.job_list = []
        self.queue_list = []
        self.processing_job = None
        self.finished_jobs = []
        self.job_count_hist = []
        
    def run(self, simulation_time):
        print("\nTime: 0 sec, Simulation starts for λ=" + str(self.arrival_rate))
        
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

        sim_time = simulation_time/10000
        current_time = 0
        while sim_time <= simulation_time:
            while True:
                if ( len(self.job_list) > 0 and 
                     self.processing_job and
                     self.processing_job.service_end_time < self.job_list[0].arrival_time ):
                    if self.processing_job.service_end_time > sim_time: break
                    finished_job = self.processing_job 
                    self.processing_job = None   
                    self.finished_jobs.append(finished_job)
                    current_time = finished_job.service_end_time
                    self.time_series[current_time] = len(self.queue_list)
                    continue
                if ( len(self.job_list) > 0 and 
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
                    if ( len(self.job_list) > 0 and 
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

            self.job_count_hist.append( len(self.queue_list) + (1 if self.processing_job else 0) )
            sim_time += simulation_time/10000

        print("Total jobs: " + str(len(self.jobs)))
        return self.time_series, self.job_count_hist, self.jobs
        
# def plot_simulation_jobs_vs_t(jobs, arrival_rate, sumarize):

def I(n, x):
    h = 0.01
    old_sum = -1
    sum = 0
    j = 0
    while np.abs(sum - old_sum) > h:
        old_sum = sum
        sum += ( (x/2)**(n+2*j) ) / ( np.prod(range(1,n))*np.prod(range(1,n+j)) )
        j += 1
    return sum

def P(n, t):
    _lambda = 2
    _mu = 10
    _rho = _lambda / _mu

    h = 0.01
    old_sum = -1
    sum = 0
    j = n+2
    while np.abs(sum - old_sum) > h:
        old_sum = sum
        sum += _rho**(-1/2) * I(j, 2*t*np.sqrt(_lambda*_mu))
        j += 1

    np.exp( -(_lambda+_mu)*t ) * ( 
        _rho**(n/2) * I(n, 2*t*np.sqrt(_lambda*_mu)) +
        _rho**((n-1)/2) * I(n+1, 2*t*np.sqrt(_lambda*_mu)) +
        (1-_rho)*_rho**n * sum
    )

if __name__ == '__main__':
    results = {}
    for arrival_rate in LAMBDAS:   
        system = System(arrival_rate, MU)
        time_series, job_count_hist, jobs = system.run(TOTAL_SIMULATION_TIME)

        # x = np.array([x for x in result])
        # y = np.array([result[x] for x in result])
        # plt.step(x,y, 'g^--', where='post')
        # plt.show()   
        plt.hist(job_count_hist, density=True)
        rho = arrival_rate/MU
        testx = np.linspace(min(job_count_hist), max(job_count_hist), num=50)
        testy = [(1-rho)*rho**n for n in testx]
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