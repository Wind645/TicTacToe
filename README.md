# TicTacToe
Finally done!
I got so excited as this model achieves an amazing result! And this is the first project I made that
has to do with reinforcement learning.

Let me first explain my design for the project here, the goal here is to train an agent that can play
tictactoe(井字棋). I did not do the tabular learning, cause I asked copilot to directly help me generate
a demo in tabular case, and I have trained that. But the agent finally turns out to be rather stupid, 
a friend of mine remarked it as 'High EQ'. 

The project here adopts DQN, a combination of RL and DL. I have manually finished the code of the environment,
dqn_agent, and the training part. And copilot generated the GUI and the remote training part.

For the concrete architecture of the agent, I used a neural network(simple FFNNs) to estimate Q value with the 
input of the state, the neural network will receive the state of the chess board and give out 9 Q values of all the actions.

Key ideas about DQN is the fixed target Q value and the replay buffer, actually they are easy to implement in the code.
The replay buffer is actually a place where experiences can be stored, and the agent can randomly sample from it to update
its weight, you can refer to the dqn_agent.py.

About the idea of the fixed target Q value, two seperate networks are needed, one as the fixed network with fixed weights,
another is updated dynamically during training. And the fixed network copies the weights from another one at intervals. The 
'update' method in dqn_agent.py implemented this part.

While training, I created two agents, and they played with each other for a hundred thousand times. so here I have two files that stores their
weights. You can directly load the in the GUI part.

If you want to try to fight with the agent, you can run gui.py and load any of the two files.
