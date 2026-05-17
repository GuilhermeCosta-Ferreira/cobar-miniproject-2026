# Wind

## Sensor Data

In the simulation, in order to obtain a signal from the effect of the wind on the fly's body, we use the function `sim.get_antenna_data(fly.name)`.

This function outputs something like the following:
```
{'l': 
    {'qpos': array([ 0.99784931, -0.06431949,  0.01159979,  0.00502095]),
    'qvel': array([ 0.45647668,  0.56760959, -0.01446059]),
    'qacc': array([-16.00091514, -22.15418647,   4.73208243]),
    'qfrc_passive': array([ 8.30671971e-04, -7.99964665e-04, -8.61428296e-05])},
 'r': 
    {'qpos': array([ 0.99932577,  0.03106377,  0.01879454, -0.00546067]),
    'qvel': array([0.44984093, 0.37956173, 0.09994724]),
    'qacc': array([-13.28875976, -13.45844387,  -3.48936196]),
    'qfrc_passive': array([-1.07141939e-03, -7.55671397e-04,  9.29898148e-06])}
}
```

Where:
- `l` and `r` are respectively the left and right antennae.
- `qpos` is the quaternion that pertains to the position of the ball joint connecting the antenna to the body. In MuJoCo, this is represented as $[w, x, y, x]$, where $w$ is a scalar part.
- `qvel` is how fast the antenna's joints are rotating (angular velocity).
- `qacc` is the angular acceleration of the antennae.
- `qfrc_passive` is the part of the applied forces coming from the passive material of the antennae. It opposes deflection.

## Wind Direction from Sensor Data

In order to calculate the wind direction from the sensor data, a potential approach is to:

1. Convert the qpos of the antenna joints into antenna "deflection vectors"
    
    i) Get rotation of each antenna relative to fly's body ("subtract" the rotation of the fly in the world from the rotation of the antennae in the world)

2. Calculate the wind direction from the 2 deflection vectors

    i) Take into account stereo effects

## Wind Magnitude from Sensor Data
