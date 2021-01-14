//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 2006-2015 OpenSim Ltd.
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#include "Source.h"
#include "Job.h"

Define_Module(Source);

void Source::initialize()
{
    // Signals
    createdSignal = registerSignal("created");

    // Parameters
    numJobs = par("numJobs");
    jobName = par("jobName").stringValue();
    if (jobName == "")
        jobName = getName();
    startTime = par("startTime");
    stopTime = par("stopTime");

    jobCounter = 0;
    WATCH(jobCounter);

    // schedule the first message timer for start time
    scheduleAt(startTime, new cMessage("newJobTimer"));
}

void Source::handleMessage(cMessage *msg)
{
    ASSERT(msg->isSelfMessage());

    if ((numJobs < 0 || numJobs > jobCounter) && (stopTime < 0 || stopTime > simTime())) {
        // reschedule the timer for the next message
        scheduleAt(simTime() + 1/par("arrivalRate").doubleValue(), msg);

        Job *job = createJob();
        send(job, "out");
    }
    else {
        // finished
        delete msg;
    }
}

void Source::finish()
{
    emit(createdSignal, jobCounter);
}

Job *Source::createJob()
{
    char buf[80];
    sprintf(buf, "%.60s-%d", jobName.c_str(), ++jobCounter);
    Job *job = new Job(buf);
    job->setKind(par("jobType"));
    job->setPriority(par("jobPriority"));
    return job;
}
