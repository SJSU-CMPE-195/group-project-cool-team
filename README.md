Group Members: 
1. Jason Hernandez (jason.a.hernandez@sjsu.edu)
2. David Zhang (david.zhang@sjsu.edu)
3. Ricky Liang (ricky.liang@sjsu.edu)
4. Viet-Joshua Le (viet-joshua.le@sjsu.edu)

Group Advisor:
Eric Vanuska (eric.vanuska@sjsu.edu)

We are S.A.F.E.R (Smart Autonomous Facility Enforcement Robot), a hardware-focused team designing an autonomous robot capable of navagating predefined paths to scan personnel and make sure they are wearing the appropriate PPE required for protocol. This project consists of a multitde of different areas, but we will be mostly working on machine learing and embedded systems.


# S.A.F.E.R (Smart Autonomous Facility Enforcement Robot)

## Team
- Jason Hernandez (jahern03)

## Project Description
S.A.F.E.R. is an autonomous mobile robot designed to automate safety inspections in construction and manufacturing environments. Using on-device computer vision, the robot patrols a designated area to identify personnel and verify compliance with PPE requirements, such as hard hats, high-visibility vests, and safety glasses.

## Proof of Concept Scope
This PoC simply demontstates the PPE recognition of the construction area, specifically the vest and hardhat items. This is to test whether or not our idea is viable. Because our model is able to differentiate a person from their respective PPE

## Prerequisites
We currently have the model + dataset listed on the github, but we are not finished with finding all the datasets in order to get all category regonition areas going. This includes PPE listed on the personnel, and the actual PPE categories themselves, requiring us to create another dataset.

## Installation
Installation can be found under "src" and "tests", where the folders show the model working as tested on a specific image. While it is not fully ready to recognize all 5 items, we are actively creating a program designed to combat this issue.

## Running the PoC
[How to run and see it working]

## Demo
[Screenshots, GIFs, or video link showing the PoC working]

## Technical Stack
- Kaggle (dataset)
- Jupiter Notebook
- GitHub
- YoloV8 Model

## What's Next (195B)
Next sememster we will focus on creating the hardware portion of the robot, and essentially bring everything together. This includes the framework, the different modules, and actual objection detection to alert if there is a hazard in the way. By the end of this semester, we plan on finishing the facial recognition for PPE detection, along with the correct datasets. 
