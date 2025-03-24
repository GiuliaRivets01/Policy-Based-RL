
# Reinforcement Learning 2024, Assignment 3: Policy-based RL

### Requirements
For all of the packages needed to run this assignment, see file *requirements.txt*, which can be
downloaded by running 

*pip install -r requirements.txt*

Personally, we haven't encountered problems while installing the
packages, but we have noticed that some people may get
problems in installing Box2D. If you encounter this problem,
a solution could be this one:

*py -m pip install --upgrade pip setuptools wheel*

### REINFORCE

The whole implementation of the REINFORCE algorithm is 
done in file *REINFORCE.py*, containing the network architecture
as well as the algorithm. 

To run the file and get the plots we have obtained in the
experiments part of the report, run the file *experiments_Reinforce.py*

The file *reinforce_tuning.py* contains the code for the
hyperparameter tuning part

### Actor Critic

The whole implementation of the Actor Critic algorithm is 
done in file *actor_critic.py*, containing the network architecture
as well as the algorithm. All of the different actor critic
implementations (namely actor critic with bootstrap and baseline
subtraction, actor critic without baseline subtraction and actor 
critic without bootstrap) are implemented in that file.

To run the file and get the plots we have obtained in the
experiments part of the report, run the file *experiments_actor_critic.py*
To enable/disable the usage of bootstrap or baseline
subtraction, you can change the value of the Boolean 
variables *use_bootstrap* and *use_baseline*

The file *actor_critic_tuning.py* contains the code for the
hyperparameter tuning part executed for the actor critic implementations


We have noticed that by running the algorithm for several subsequent repetitions
sometimes it may happen to receive a Nan value from the network, 
probably due to exploding/vanishing gradients. We have done some research and
discovered that it is a common issue encountered with the actor critic
algorithm and most of the solutions we have found on the Internet suggested 
to use gradient clipping.
However in our case simply using gradient clipping didn't work, and therefore
we have addressed the issue by running again the 
that particular experiment if such an error was encountered. 




