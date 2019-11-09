# Web Crawler with PPO and Graph Neural Network
This is an experimental project on trying to make a webcrawler that attempts to find a target webpage on a Wikipedia although
it could be extended to many other websites and even the internet in general. 
As it does this it builds up a graph of the pages that it visits and uses this to asses which page is should explore
each step. Right now whenever the agent chooses a page to explore all outlink pages on the chosen page are added to the graph the agent
is building up. A more efficient implementation would make a second model that picks which outlink to explore given the
page it chooses so that it could work on pages that have a lot of hyperlinks.

This is still in progress, as of right now it does not perform very well on sparser graphs but it does perform well
on denser graphs. There are a lot of possible improvements to be made from hyperparam tuning to GNN arch. improvements.

To run, run `main.py`. Which will either use `consants/constants.json` or `constants/constants-grid.json` depending on what
is run in the main function.

Currently, two baseline models are supplied in `models`, `fully_connected.py` which emulates a fully connected graph and
`no_structure.py` which emulates no graph and instead each neural net model is just applied to each node discovered.
The two main models are from deepmind: _Graph Networks as Learnable Physics Engines for Inference and Control_ and
NerveNet:  _NerveNet: Learned Structure Policy with Graph Neural Networks_. The main difference is that deepmind 
allows for edges to have features and nervenet does not. Both are plausible here. 

All of the wikipedia dumps that I converted to graphs are in `data`. There are some interesting jupyter notebooks in
`notebooks` that show my progress throughout this project (I started by learning graph neural networks and coding
them in these notebooks and then transferring to normal python files). 