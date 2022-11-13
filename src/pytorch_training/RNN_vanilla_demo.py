#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:15:14 2022

@author: benedict

PyTorch demo implementing an RNN

RNNs are NN that take an input, do some internal operations (in the neuron) and get hidden states, that are fed back into the network as inputs
In essence, they can use previous knowledge to udpate the current state compared to a normal feed forward neural network that has no 'memory' of previous steps
Check out Andrej Karpathy article on RNNs
RNNs can have one to many, many to one and many to many (seq2seq?) relationships between input and output

There is the process illustrated as 'unfolding' the network in order to see how the hidden state carries across all the processing

Advantages:
        Possible to process input of any length
        Model size not increasing with size of input (like a feed forward network)
        Computation takes into account historical information
        Weights are shared across time
Disadvantages
        Slow computation
        Difficulty accessing information from a long time ago
        Cannot consider any future input for the current state (one direction only, not bidirectional)

Task at hand:
    We want to do name classification 
    Different files with names from different countries (last names)
    Detect from which country the name is from
    Take the whole name as a sequence and use each single letter as one input for the RNN
    For this we need the helper functions (utils.py)
 
RNN structure:   
                     __
input\         /hidden \
      combined        / 
i2O/         \i2h   |
 \              |  |
softmax        |  /
  \           |  /
output       hidden

"""

import torch
import torch.nn as nn
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS # attributes, or variables
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example # methods or functions

torch.cuda.is_available() # check if gpu is available

# building from scratch to override the RNN that's already in PyTorch for learning purposes
class RNN(nn.Module): # inherit from nn.Module
    
    def __init__(self, input_size, hidden_size, output_size): # creates the structure of the network with no real values in it
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        # nn.Linear creates a fully connected layey with no activation function, the activation function must be applied to this layer seperately in the forward method
        #self.i2x = nn.Linear(in_features=, out_features=)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # combined to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # combined to output
        self.softmax = nn.LogSoftmax(dim=1) # 1, 57 -> we need the second dimension i.e. 57
        
        
    def forward(self, input_tensor, hidden_tensor): # this is different to a feed forward NN where there is only one input, now we input the hidden state as well
        # combine the input and hidden tensors, process them through the linear layers and then softmax one for output and return both output and hidden
        
        combined = torch.cat((input_tensor, hidden_tensor), 1).to('cuda')
        
        hidden = self.i2h(combined) # process the input and hidden tensors through the fully connected layer
        output = self.i2o(combined) # process hidden output through the fc layer
        output = self.softmax(output) # apply activation function
    
        return output, hidden
    
        
    def init_hidden(self): # need initial hidden state in the beginning
        return torch.zeros(1, self.hidden_size)



category_lines, all_categories = load_data()
n_categories = len(all_categories) # number of classes for this classification task which is 18
n_hidden = 128 # hyperparameter to be tuned


rnn = RNN(N_LETTERS, n_hidden, n_categories).to('cuda') # running the __init__() method to create the structure of the network, nothing has happened yet
#for p in rnn.parameters(): # parameters are the weights and biases for each layer!
    #print(p.grad) # torch.Size([n_nodes, n_inputs]) = n_nodes * n_inputs = n_weights, next is number of biases

"""
whole sequence/name, of steps:
treat our name as one sequence of characters and each character is one input
repeatedly apply the RNN to all the characters in the name and then at the very end 
we take the last output and apply the softmax and take the one with the highest probability

"""

input_tensor = line_to_tensor('Albert')
print(input_tensor.size())
print(input_tensor[0]) # this is 'A'
category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
print(line_tensor[0].size())

hidden_tensor = rnn.init_hidden()
print(hidden_tensor.size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(N_LETTERS, n_hidden, n_categories).to(device)
print(model)






# =============================================================================
# output, next_hidden = rnn(input_tensor[0], hidden_tensor) # we need to repeatedly call this through all the examples of the sequence
# =============================================================================

def category_from_output(output): # likelihood of each category, so we are going to return the index of the highest probablity
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


"""
Training for this example: 
We have a word going on that like a batch, we are processing each character in the word individually as input to the network
The network keep iterating through the characters (each is one hot encoded into a 1, 57 tensor)
When the end of the word (or sequence, as the word in total is the sequence) is reached, then the loss is calculated on the 
final output from the sequence
"""

# criterion is any loss function, we use this to calculate the error with respect to the weights and biases
criterion = nn.NLLLoss() # negative likelihood loss
learning_rate = 0.010
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) # optimizer is linked to the models parameters here during instantiation

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    
    # forward propagation
    for i in range(line_tensor.size()[0]): # process a single sequence in this for loop
        output, hidden = rnn(line_tensor[i].to('cuda'), hidden.to('cuda')) # hidden state is being updated each iteration as is the output // forward passes!

#     for p in rnn.parameters(): # parameters are the weights and biases for each layer!
#         print(p.grad) # torch.Size([n_nodes, n_inputs]) = n_nodes * n_inputs = n_weights, next is number of biases
#     params = sum([np.prod(p.size()) for p in rnn.parameters()])
#     print(params) # sum of params
#     now the forward pass has been completed for a whole word, with the output at the last step of the sequence (last letter) being what we calculate loss on
#     each character at a time, the loss is calculated and the gradient is updated
    
    optimizer.zero_grad() # zero the parameters of the gradient so they don't accumulate
    
    print(category_tensor)
    
    loss = criterion(output, category_tensor) # calculate the loss based on the error function
    torch.autograd.set_detect_anomaly(True)
    # backward propagation -> backward pass that calculates the gradients 
    loss.backward() # loss with respect to the parameters is calculated which is the gradient! 
    optimizer.step() # update the parameters based on their grad attribute using the step() function
     
    return output, loss.item() # loss.item() returns a float value as opposed to a tensor






# =============================================================================
# now we train
# =============================================================================
    
current_loss = 0
all_losses = []

total_correct = 0

plot_steps, print_steps = 1000, 1000
n_iterations = 100000

for i in range(n_iterations):
    category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
     
    output, loss = train(line_tensor.to('cuda'), category_tensor.to('cuda'))
    current_loss += loss
    
    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps) # taking averages
        current_loss = 0
        
    if (i + 1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
#         print(f'{i} {i/n_iterations*100} {loss:.4f} {line} / {guess} {correct}')
        print(f'{i} {loss:.4f} {line} / {guess} {correct}')
        
        
category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)
print(category_tensor)        
    
        
plt.figure()
plt.plot(all_losses)
plt.show()

rnn.eval()
torch.save(rnn.state_dict(), 'models/rnn.pt')

for param in rnn.parameters():
    print(param)

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()
        
        # forward propagation
        for i in range(line_tensor.size()[0]): # process a single sequence in this for loop
            output, hidden = rnn(line_tensor[i].to('cuda'), hidden.to('cuda')) # hidden state is being updated each iteration as is the output // forward passes!

        guess = category_from_output(output)
        print(guess)



while True:
    sentence = input('Input:')
    if sentence == "quit":
        break
    
    predict(sentence)



    
     
     
    
'''
'''

"""
loss.backward()
The autograd mechanims is the process in charge of performing gradient computation (it doesn't change the gradient though!)
It allows you to call backward() on a torch.Tensor (your loss)
This process is navigating through the 'computation graph', updating each of the paremeters gradients (calculating their deltas) 
by changing their grad attribute. This means that at the end of a backward call, the networks learned parameters that were used to compute this output
have a grad attribute containing the gradient of the loss with respect to that parameter. 
This means, the loss has been calculated with respect to each parameter (or weight) according to the architecture of the network i.e. the computational graph, which contains each layer and the activation functions are imbedded here
loss with respect to the parameters is the same as error with respect to the weights!

"""

"""
optimizer.step()
The optimizer is independent of the backward pass since it doesn't rely on it
the optimizer's task is to take the paremeters of the model independently (that is irrespective of network architecture) and update them
using a given optimization algorithm. 
It goes through all the paremeters it was initialised with and updates them using their repsective gradient value 
(which is supposed to be stored in the grad attrirbute by at least one backpropagation)
"""

    
# =============================================================================
#         # zero the parameter gradients as gradients should be accumulated over a batch, if we don't zero them they will accumulate across batches
#         optimiser.zero_grad()
#         
#         # forward + backward + optimise
#         outputs = net(inputs) # forward propagate the inputs through the network
#         loss = criterion(outputs, labels) # calculate the loss between predicted and labels
#         loss.backward() #  backward pass that calculates the gradient through the network using the backpropagation algorithm
#         optimiser.step() # now we adjust the weights according to our deltas calculated in the previous line of code
#         
# =============================================================================






















































