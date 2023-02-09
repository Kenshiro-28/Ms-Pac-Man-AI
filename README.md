# Ms. Pac-Man AI

This is a Ms. Pac-Man artificial intelligence based on the T-Rex evolutionary neural network. It trains playing the game until it's unable to improve its best score. After completing the training, the program plays a last game rendering the game screen.

## Game information

Ms. Pac-Man is a 1982 maze arcade video game developed by General Computer Corporation and published by Midway. It is the first sequel to Pac-Man (1980) and the first entry in the series to not be made by Namco. Controlling the title character, Pac-Man's wife, the player is tasked with eating all of the pellets in an enclosed maze while avoiding four colored ghosts. Eating the larger "power pellets" lets the player eat the ghosts, who turn blue and flee. 

Ms. Pac-Man was acclaimed by critics for its improvements to the original gameplay and for having a female protagonist; some have described it as superior to Pac-Man. It has been listed among the greatest video games of all time and as one of the most successful American arcade games ever made. The game's success inspired a variety of successful merchandise, several ports for numerous home consoles and handheld systems, a television cartoon that included Pac-Man, and numerous video game sequels and remakes that spawned a Ms. Pac-Man video game spin-off series. The rights to the game are owned by Namco's successor company, Bandai Namco Entertainment. 

This program uses the Atari 2600 console emulator provided by Gymnasium.

## Neural network information

T-Rex is an evolutionary neural network. It learns by adjusting the strength of the connection weights by mutation and selection. The programmer must define the problem to solve with a scoring system so that T-Rex can evolve gradually until finding the optimal solution.

Main features:

- Binary feedforward neural network
- Configurable number of inputs, hidden layers and outputs
- Developed using object-oriented programming
- Fast, robust and portable

### Input layer

The input of the neural network is the state of the 128 bytes of RAM of the Atari 2600. The input layer has 1024 neurons.

### Hidden layer

The neural network has 2 hidden layers. The T-Rex architecture states that the number of neurons in each hidden layer is set as the number of input neurons, so it has 1024 neurons.

### Output layer

The neural network outputs are the button activations on the player's gamepad. As the game emulation provides 9 possible actions, the output layer has 4 neurons.

## Installing dependencies

### T-Rex

You must compile T-Rex as a shared library:

https://github.com/Kenshiro-28/T-Rex

Copy the generated file **libT-Rex.so** in the folder /usr/local/lib

Run this command to add the folder to the library path:

```
$ sudo ldconfig /usr/local/lib
```

Copy the header files in the folder /usr/local/include/T-Rex

``` 
T-Rex
   ├── data_tier
   │   └── DataManager.h
   ├── logic_tier
   │   ├── NeuralLayer.h
   │   └── NeuralNetwork.h
   └── presentation_tier
       └── ConsoleManager.h
```

### Python

Run this command to install Python:

```
$ sudo apt install python3-dev python3-pip python3-wheel
```

### Gymnasium

Run this command to install Gymnasium with the Atari 2600 dependencies:

```
$ pip3 install gymnasium[atari]
```

Run this command to install AutoROM and download the ROMs:

```
pip3 install gymnasium[accept-rom-license]
```

## Running

Run this command in the root folder to start the program:

```
$ python3 ms-pacman.py
```

This project includes a trained neural network. Running the program will skip the training phase if there is a neural network file in the same folder. To train a new neural network, delete the file **neural_network.json**.

----- Work in progress -----

