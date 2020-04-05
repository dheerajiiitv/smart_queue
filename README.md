 *Template for Your ReadMe File*
 
 A ReadMe file helps reviewers in accessing and navigating through your code. This file will serve as a template for your own ReadMe file. Do fill out the sections carefully, in a clear and concise manner
 
 # Smart Queue Monitoring System
*TODO:* Write a short introduction to your project.

In this project I will be building an application to reduce congestion in queuing systems by guiding people to the least congested queue. Queue problem exists in many areas, and out this areas we are focussing on three main areas - 
1. Transportation
2. Retail
3. Manufacturing

With the help of Artificial intelligence we will try to optimize number of people in queues so avg. turnarounf time per customer on a retail counter, or company production belt to save time and increase profit, production etc. 
To solve this problems we will be using IoT devices for real time monitoring and faster inference, but for each areas inference time, money, space constraint are available. So we will try IoT device with differnt sets of hardware such as GPU, VPU, FPGA and will find out which suits us best.


## Proposal Submission
*TODO* Specify the type of hardware you have chosen for each sector 
Mention the type of hardware you have chosen for each of the scenarios:
- Manufacturing:
    - Device type - *CPU + FPGA*
    - Reason:
       - Programmable - FPGA are programmable so you program it according to your requirements
       - Power Requirements - Power requirement is high, but Mr Vishwa does not have any problem with it.
       - Space Requirements - Space requirement is less as compared to hardware
       - Economic Constraints - Cheap as they are programmable, compared to other hardware

- Retail:
    - Device type - *CPU + GPU*
    - Reason:
       - Programmable - Non Programmable
       - Power Requirements - Power requirement is high, but supports multiple threading helps in high inference time.
       - Space Requirements - Very low as compared to FPGA and VPU
       - Economic Constraints - Expensive compared to VPU and FPGA
- Transportation: 
    - Device type - *VPU*
    - Reason:
       - Programmable - Non Programmable
       - Power Requirements - Very less as compared to other devices.
       - Space Requirements - Chip based very less space required.
       - Economic Constraints - High compared to FPGA

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.


1. Install dependencies using below command:
    - `pip install -r requirements.txt`
2. open "people_deployment-manufacturing.ipynb"
3. Execute cells in order.
4. Results can be found in folder - `results/manufacturing/<device_name>`
5. Than do the same for the other two notebook.


## Documentation
*TODO* In this section, you can document your project code. You can mention the scripts you ran for your project. This could also include scripts for running your project and the scripts you ran on Devcloud too. Make sure to be as informative as possible!
1. person_detect.py - This contain code to load the model, make infernce on it and return result
2. person_detect_job.sh - This file will be created after you execute cell in the first jupyter notebook, this will used to execute python program to get inference. 
3. *.ipynb - These are notebook will help you to explore different hardware devices to run inference and find comparision b/w them.

## Results
*TODO* In this section you can write down all the results you got. You can go ahead and fill the table below.

### Application - 
#### 1. Manufacturing - 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                    0.015                          |             1.6            |              FP16           |
| HETERO:GPU,CPU              |             0.026                                 |               36             |      FP16                    |
| HETERO:FPGA,CPU             |              0.013                                |            29.29                |   FP16                       |
| VPU              |                                              |                            |                         |

#### 2. Retail 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                    0.0167                          |            1.7                |           FP16              |
|  HETERO:GPU,CPU              |               0.026                               |             36.56               |    FP16                     |
| HETERO:FPGA,CPU              |                0.012                              |              30.57              |    FP16                     |
| VPU              |                                              |                            |                         |

#### 3. Transportation 

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                   0.017                       |           1.8                 |            FP16             |
| HETERO:GPU,CPU                |            0.026                                  |         35.23                   |  F16                       |
| HETERO:FPGA,CPU             |                 0.0128                             |           29.65                 |     FP16                    |
| VPU              |                                              |                            |                         |
## Conclusions
*TODO* In this section, you can give an explanation for your results. You can also attach the merits and demerits observed for each hardware in this section

I can conclude using different devices help different kind of application. Always you need to trade off b/w performance and inference time. model space and performance. So you have to try each different kind of models on different types of devices to find best suited model for us.

Due to the use HETERO inference time GPU decreases compare to CPU, as current our model is not much complex CPU performance is good compare to GPU, but if we have more complex model GPU inference time will reduce.


## Stand Out Suggestions
*TODO* If you have implemented the standout suggestions, this is the section where you can document what you have done.
- Fallback Policy: Use OpenVINO's Hetero plugin to use a fallback device like CPU when the main prediction device fails. Write about the changes you had to make in the code to implement these changes. More information regarding Fallback Policy can be found [here](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HETERO.html)
- MultiDevice Plugin: Use the multi device plugin available as a part of the OpenVINO toolkit. Write about the changes that you had to make in your code to implement this.
