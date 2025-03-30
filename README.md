# Champion
**Champion** is a reinforcement learning model intended to optimally play Spy Cards from *Bug Fables: The Everlasting Sapling*.

The goal for **Champion** is to be able to provide a sufficient challenge for players, without cheating, that other decision-making bots have failed to accomplish.

# Installation
Run the following in a terminal or command prompt. It will install the git to the directory specified in the terminal/command prompt.
```
git clone https://github.com/FoxtrotOnce/Champion.git
```
# Usage
Run the model by running PPO.py, and view the data in GraphPPO.py.

# Progress
- [x] Create a Spy Cards environment accurate to *Bug Fables* for the model to interact with.
- [x] Set up modeling framework and extra classes/functions for testing.
- [x] Get >80% win-rate on a playing model using a Leafbugs deck VS. *Bug Fables* bot using a Thugs deck.
- [ ] Get >80% win-rate on a playing model using any meta deck VS. *Bug Fables* bot using any meta deck.
- [ ] Change modeling framework to make the playing model work with a model that predicts the opponent's move.
- [ ] Get >80% win-rate with the combined models using any meta deck VS. *Bug Fables* bot using any meta deck.
- [ ] Get data from the Spy Cards Tournament to use for supervised learning of the opponent prediction model, to predict human players.
- [ ] Get >80% win-rate with the combined models using any deck VS. a human player using any deck.

# License
**Champion** is an open-sourced software licensed under the [MIT license](https://opensource.org/license/MIT "MIT license").
