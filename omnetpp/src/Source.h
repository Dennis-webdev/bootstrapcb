//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 2006-2015 OpenSim Ltd.
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#ifndef __QUEUEING_SOURCE_H
#define __QUEUEING_SOURCE_H

#include "QueueingDefs.h"

class Job;

/**
 * Generates jobs; see NED file for more info.
 */
class Source : public cSimpleModule
{
    private:
        // Signals
        simsignal_t createdSignal;

        // Parameters
        int numJobs;
        std::string jobName;
        simtime_t startTime;
        simtime_t stopTime;

        int jobCounter;

    protected:
        virtual void initialize() override;
        virtual void handleMessage(cMessage *msg) override;
        virtual void finish() override;
        virtual Job *createJob();
};

#endif


