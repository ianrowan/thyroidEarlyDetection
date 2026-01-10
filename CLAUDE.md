# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I have hyperthroidism. It is often difficult to adjust dosage of medication etc between blood tests. I've tracked all apple watch/whoop data for years however and can cross reference periods of time where I was different degrees of hyper thyroid or trending towrds it. 

I would like to research and build/train ML models to find small patterns in a variety of data(RHR, HRV, Respitory rate, sleep etc) that indicate beginning current or end of hyperthyroid periods.

1. Initial Research Phase: 
- create a research plan
- understand data structure of available input data(assume apple health output data with >3 years of all data tracked by apple watch/whoop)
- There will be some subjective/ambiguous output data that we should make a decision on how to use/label training sets. I have both thyroid labs(T3, TSH, T4) from recent times and can label period from memory in the past
- All ML/statictical methods are on the table so we will need to research this
- Research of existing methods(if any) similar to this exist
- Results of this phase should include:
  - Understanding of data structure
  - A plan on how to create data set(including labeling efficiently)
  - proposed approachs for ML(can include multiple that we will benchmark in phase 2)
  - Fill in (Requests of user) section with request data/action
  - Research output to research.md
2. Implementation Phase
- should include data labelling activities 
- implementing proposed ML models/data pipelines from Phase 1
- Should implement robust experiment tracking infrastructure 
- implementation should be relatively portable across machines/environements
- use python and most effective packages for data/ML 
- Output:
  - data should be labeled/pipelines created to automate
  - ML model training should run effectively 
  - ready for multi-approach experimentation 
3. Experimentation Phase
- Run all ML approaches and record benchmarking data
- Include hyperparam tuning for all models
- run this process iteratively 
- Analyze results and create an md file writing up summary of results 
- Add to Request of User if new approaches need to be signed off on
- Output:
  - Successful runs of all Model/params
  - benchmark data in some file
  - ability to make decicision on the correct approach 
4. Productionization/infernce Phase
- Use chose approach from prior steps to build an infernece pipeline that can be integrated to some app(seperate project) to create the output predictions 
- Output:
  - local command line version of inference pipeline
  - approach or packages to use in other project 

Approach and ideas
These are just ideas and dont need to be incorporated if not effective
- Use timeseries approaches to use more then just the current data to predict states(ie trailing propobailities)
- Use trends in output to weight towards future output
- Above I'm basically saying to be baysian 
- Output current state and forcases

Hardware Access
- Local dev is on a 16GB macbook m3 pro
- Access locally to a titan RTX with 24Gb vram
- Open to using cloud services if necessary but only for vram or necessary speed up

## Always Rules 

- Respond concisely: only relevant info, code, plans
- Think relatively deeply: contrary to concise response think alot
- Do not comment code
- Flag strange patterns in data and add to request of user
- Programming language: python

## Workflow preference

- Plan -> confirm -> code -> test -> iterate
- commit changes prior to new tasks and refactors
- use version control smartly 

## Development Commands

*To be added as the project develops.*

## Architecture

*To be documented as the codebase is built.*

## Requests of user

*Use this section to request dataset, etc from the user, use it like a checklist and checkoff once you receive data*

[ ] Apple Health export (full XML from iPhone Health app → Profile → Export All Health Data)
[ ] Labeling session: date ranges + severity (Normal/Mild/Moderate/Severe) for remembered hyperthyroid episodes
[ ] Lab results with dates (TSH, T3, T4 values for the 6 known lab draws)
[ ] Medication history with dates (optional, for validation/context)
