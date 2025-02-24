# Champion
**Champion** is a reinforcement learning model intended to optimally play Spy Cards from *Bug Fables: The Everlasting Sapling*.

The goal for **Champion** is to be able to provide a sufficient challenge for players, without cheating, that other decision-making bots have failed to accomplish.

# Installation
Run the following in a terminal or command prompt. It will install the git to the directory specified in the terminal/command prompt.
```
git clone https://github.com/FoxtrotOnce/Champion.git
```
# Usage
Configure the playing model's hyperparameters in config.py.
You can modify the playing model's architecture in DQN.py by changing layers in the create_model function, or changing the loss/optimizer.
```python
def create_model(input_shape: tuple, output_shape: int) -> tf.keras.Model:
    """Create and compile a new TensorFlow model."""
    input_layer = tf.keras.layers.Input(input_shape)
    example_dense = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    example_mha = tf.keras.layers.MultiHeadAttention(3, 18)(example_dense, input_layer)
    ...
    output_layer = tf.keras.layers.Dense(output_shape, activation='linear')(last_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mse'
    )
```
Run the model by running DQN.py.

# Progress
- [x] Create a Spy Cards environment accurate to *Bug Fables* for the model to interact with.
- [x] Set up modeling framework and extra classes/functions for testing.
- [ ] Get >80% win-rate on a playing model using a Leafbugs deck VS. *Bug Fables* bot using a Thugs deck.
- [ ] Get >80% win-rate on a playing model using any meta deck VS. *Bug Fables* bot using any meta deck.
- [ ] Change modeling framework to make the playing model work with a model that predicts the opponent's move.
- [ ] Get >80% win-rate with the combined models using any meta deck VS. *Bug Fables* bot using any meta deck.
- [ ] Get data from the Spy Cards Tournament to use for supervised learning of the opponent prediction model, to predict human players.
- [ ] Get >80% win-rate with the combined models using any deck VS. a human player using any deck.

# License
**Champion** is an open-sourced software licensed under the [MIT license](https://opensource.org/license/MIT "MIT license").
